import { MuseClient, zipSamples } from 'muse-js'
import App from '../App'

// ── Constants ─────────────────────────────────────────────────────────────────

const PPG_FS       = 64
const PPG_INFRARED = 1          // ppgChannel 1 = infrared, best cardiac signal
const PPG_WIN_SAMP = PPG_FS * 6 // 6-second MSPTDfast analysis window = 384 samples
const PPG_RUN_STEP = PPG_FS     // re-run MSPTD every ~1 s of new data

// Online IIR bandpass (HP 0.5 Hz + LP 3.5 Hz) — covers 30–210 BPM
// First-order high-pass: α = 1 / (1 + 2π·fc/fs)
const HP_ALPHA = 1 / (1 + (2 * Math.PI * 0.5) / PPG_FS)    // ≈ 0.953
// First-order low-pass: α = 2π·fc/fs / (1 + 2π·fc/fs)
const LP_ALPHA = ((2 * Math.PI * 3.5) / PPG_FS) / (1 + (2 * Math.PI * 3.5) / PPG_FS) // ≈ 0.255

// MSPTDfast v2 downsampling (ds_freq = 20 Hz)
const DS_FACTOR = Math.floor(PPG_FS / 20)   // 3 — decimation factor
const DS_FS     = PPG_FS / DS_FACTOR        // ≈ 21.33 Hz for analysis

// Plausible heart rate bounds
const HR_MIN = 30    // bpm
const HR_MAX = 200   // bpm

// Peak refinement tolerance after upsampling (tol_durn = 0.05 s, ds_fs ≥ 20 Hz)
const REFINE_TOL    = Math.ceil(PPG_FS * 0.05)   // 4 samples
// Refractory: reject peaks closer than 300 ms (mirrors MIN_PEAK_DIST in original)
const MIN_PEAK_DIST = Math.round(0.3 * PPG_FS)   // 19 samples

// Accelerometer head-pose smoothing
const ACC_ALPHA = 0.08           // low-pass coefficient — keeps only slow head movements

// EEG spectral analysis
const EEG_FS       = 256
const EEG_BUF_SIZE = 256

// Morlet wavelet parameters (following BOSC_tf.m convention).
// Delta (1–4 Hz) uses a sparse DFT because the ±3σ wavelet window at 6 Hz already
// reaches 245 samples — there is no room left for lower frequencies in a 256-sample buffer.
const WAVELET_WAVENUMBER = 6   // τ (cycles per Gaussian σ); higher → better freq, worse time
// Theta uses a reduced wavenumber (4 instead of 6) so the kernel fits comfortably
// in the 256-sample buffer: halfWin drops from 123→82, giving 92 valid samples
// instead of 10. The broader spectral smearing is acceptable for the 4–8 Hz band.
const WAVELET_WAVENUMBER_THETA = 4

// Representative center frequency per band for wavelet power estimation (theta and above).
const WAVELET_BAND_FREQS = { theta: 6, alpha: 10, beta: 20, gamma: 40 }  // Hz

// DFT bins needed for delta (1 Hz resolution at 256 Hz / 256 samples → bin k = k Hz).
const DFT_BINS = [1, 2, 3]   // 1–4 Hz, upper bound exclusive

// Aperiodic background model: log₁₀(P) = a + b·log₁₀(f)
// Fitted from quality-weighted average wavelet powers at the theta–gamma center frequencies.
// Delta normalization is extrapolated to 2 Hz using the same model.
const AP_FIT_FREQS = [6, 10, 20, 40]               // Hz — must match WAVELET_BAND_FREQS values
const AP_FIT_BANDS = ['theta', 'alpha', 'beta', 'gamma']

// Re-fit the aperiodic model every N analysis windows (50% overlap → 128 new samples/window)
const APERIODIC_UPDATE_INTERVAL = 10    // ≈ 5 s between refits
const AP_SMOOTH                 = 0.3  // EMA coefficient for incremental model updates

// Minimum number of aperiodic model refits before band power output starts.
// Ensures the model has converged before values reach the shaders.
const AP_MIN_REFITS = 3

// EMA smoothing on final bandPower output (~2 Hz update rate).
// Lower = smoother (longer settling), higher = faster tracking.
// 0.35 at 2 Hz ≈ 1.5 s settling time — matches perceptual EEG state timescale.
const BAND_SMOOTH = 0.35

// Display ring-buffer lengths
const EEG_DISPLAY_LEN = 256 * 4    // 4 s of raw EEG at 256 Hz
const IMU_DISPLAY_LEN = 52 * 4     // 4 s of IMU at ~52 Hz

// Spectrogram (Hann-windowed DFT, quality-weighted channel average)
const SPEC_BINS        = 50        // frequency bins 1–50 Hz (1 Hz/bin at 256 Hz / 256 samples)
const SPEC_COLS        = 280       // rolling history columns (matches canvas pixel width)

// High-resolution low-frequency spectrogram (0.5–8.0 Hz at 0.1 Hz resolution)
// Requires a 2560-sample (10 s) buffer: Δf = Fs/N = 256/2560 = 0.1 Hz
const SPEC_LO_BUF_SIZE = 2560
const SPEC_LO_MIN_HZ   = 0.5
const SPEC_LO_MAX_HZ   = 8.0
const SPEC_LO_STEP_HZ  = 0.1
const SPEC_LO_NUM_BINS  = Math.round((SPEC_LO_MAX_HZ - SPEC_LO_MIN_HZ) / SPEC_LO_STEP_HZ) + 1  // 76

// Signal quality RMS thresholds (µV, after mean subtraction)
const SQ_WIN  = 256   // samples per RMS window
const SQ_LOW  = 50    // below → 'good'
const SQ_HIGH = 100   // above → 'poor'; in between → 'marginal'

// Default channel-quality aggregation settings
const DEFAULT_BAD_CH_THRESHOLD = 'poor'   // 'poor' | 'marginal'
const DEFAULT_MARGINAL_WEIGHT  = 0.5      // weight for 'marginal' channels (0–1)

// ── MSPTDfast v2 helpers ───────────────────────────────────────────────────────

/** Remove best-fit linear trend from a signal (mirrors MATLAB detrend). */
function _detrend(sig) {
  const N = sig.length
  if (N < 2) return sig.slice()
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0
  for (let i = 0; i < N; i++) {
    sumX += i; sumY += sig[i]; sumXX += i * i; sumXY += i * sig[i]
  }
  const denom = N * sumXX - sumX * sumX
  if (denom === 0) return sig.slice()
  const slope     = (N * sumXY - sumX * sumY) / denom
  const intercept = (sumY - slope * sumX) / N
  return sig.map((v, i) => v - (slope * i + intercept))
}

/**
 * MSPTDfast v2 core detector (port of msptdpcref_beat_detector).
 * Detects peaks and onsets in a short PPG window via multi-scale scalograms.
 *
 * @param {number[]} sig     — bandpass-filtered PPG samples
 * @param {number}   fs      — sampling frequency of sig (Hz)
 * @param {number}   minHrHz — lower HR bound in Hz (default 30 bpm → 0.5 Hz)
 * @returns {{ peaks: number[], onsets: number[] }} — 0-based indices
 */
function _msptdDetect(sig, fs, minHrHz = HR_MIN / 60) {
  const N = sig.length
  if (N < 4) return { peaks: [], onsets: [] }

  const L      = Math.ceil(N / 2) - 1
  const durn   = N / fs

  // Limit scales to those representing plausible HRs (use_reduced_lms_scales=true)
  // Scale k corresponds to frequency (L/k)/durn Hz; keep where that >= minHrHz
  // → k <= L / (durn * minHrHz)
  const maxScale = Math.min(L, Math.floor(L / (durn * minHrHz)))
  if (maxScale < 1) return { peaks: [], onsets: [] }

  // Detrend (Step 1 of MSPTD)
  const x = _detrend(sig)

  // Build Local Maxima/Minima Scalograms as flat Uint8Arrays [maxScale × N]
  const mMax = new Uint8Array(maxScale * N)
  const mMin = new Uint8Array(maxScale * N)
  for (let k = 1; k <= maxScale; k++) {
    const row = (k - 1) * N
    for (let i = k; i < N - k; i++) {
      const xi = x[i], xL = x[i - k], xR = x[i + k]
      if (xi > xL && xi > xR) mMax[row + i] = 1
      if (xi < xL && xi < xR) mMin[row + i] = 1
    }
  }

  // Step 2: find scale lambda with maximum row-sum (most detections)
  let lambdaMax = 0, lambdaMin = 0, bestMax = 0, bestMin = 0
  for (let k = 0; k < maxScale; k++) {
    const row = k * N
    let sMax = 0, sMin = 0
    for (let i = 0; i < N; i++) { sMax += mMax[row + i]; sMin += mMin[row + i] }
    if (sMax > bestMax) { bestMax = sMax; lambdaMax = k }
    if (sMin > bestMin) { bestMin = sMin; lambdaMin = k }
  }

  // Steps 3–4: collect columns where ALL rows 0..lambda agree (intersection)
  const peaks = [], onsets = []
  for (let i = 0; i < N; i++) {
    let isPk = true, isTr = true
    for (let k = 0; k <= lambdaMax && isPk; k++) { if (!mMax[k * N + i]) isPk = false }
    for (let k = 0; k <= lambdaMin && isTr; k++) { if (!mMin[k * N + i]) isTr = false }
    if (isPk) peaks.push(i)
    if (isTr) onsets.push(i)
  }

  return { peaks, onsets }
}

export default class EEGManager {
  // Public state
  isConnected = false
  bandPower   = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
  heartRate   = 70               // BPM (initialised to resting nominal)
  heartPulse  = 0                // 0–1 oscillator synced to heartRate
  headPose    = { pitch: 0, roll: 0 }  // radians from accelerometer

  batteryLevel   = null          // 0–100 (%) or null when disconnected
  deviceName     = null          // Muse device name (for recording metadata)
  deviceInfo     = null          // muse-js deviceInfo() result (hw/fw/etc), for metadata
  onBatteryLevel = null          // optional callback(level)
  onDisconnected = null          // optional callback
  debug          = false         // set true to log pipeline internals each analysis window

  // ── Active-band filter ────────────────────────────────────────────────────
  // Only bands in this Set participate in the aperiodic-normalised relative-power
  // calculation. Bands absent from the Set are zeroed in bandPower output, so they
  // cannot dominate the normalisation. Delta is excluded by default because its
  // large, movement-prone power would otherwise swamp the higher bands; adjust
  // this Set to change which bands are shown in the EEG Bands panel.
  normalizeBands = new Set(['theta', 'alpha', 'beta', 'gamma'])

  // ── Configurable channel-quality aggregation ──────────────────────────────
  // badChannelThreshold: quality level at or below which a channel is a drop candidate.
  //   'poor'     → drop channels rated 'poor' only (default)
  //   'marginal' → drop channels rated 'poor' or 'marginal'
  // Up to 2 channels are dropped; at least 2 are always retained.
  badChannelThreshold  = DEFAULT_BAD_CH_THRESHOLD
  // marginalChannelWeight: weight assigned to 'marginal' channels in the quality-weighted
  // average of the retained channels (0 = ignore, 1 = treat as good).
  marginalChannelWeight = DEFAULT_MARGINAL_WEIGHT

  // Display buffers — raw data for live plots (populated only when connected)
  eegChannels   = [[], [], [], []]        // per-channel raw EEG, rolling EEG_DISPLAY_LEN
  accelDisplay  = { x: [], y: [], z: [] } // rolling IMU_DISPLAY_LEN
  gyroDisplay   = { x: [], y: [], z: [] } // rolling IMU_DISPLAY_LEN
  signalQuality = ['poor', 'poor', 'poor', 'poor'] // per EEG channel

  // Monotonically increasing sample counters — never reset by the rolling window
  eegSampleCount      = 0
  ppgSampleCount      = 0
  imuSampleCount      = 0
  spectrumSampleCount = 0

  // ── Private ─────────────────────────────────────────────────────────────────
  _client      = null
  _eegSub      = null
  _eegRawSub   = null
  _ppgSub      = null
  _accelSub    = null
  _gyroSub     = null
  _telemetrySub = null

  // EEG — per-channel rolling analysis buffers (EEG_BUF_SIZE samples each)
  _chBuffers    = [[], [], [], []]
  _analysisCount = 0   // new samples since last spectral analysis trigger

  // Aperiodic background model params: log₁₀(P) = a + b·log₁₀(f)
  _apModel       = { a: 0, b: -1.5 }
  _apWindowCount = 0   // analysis windows since last model refit
  _apRefitCount  = 0   // total refits performed (for warm-up gating)

  // Precomputed signal-processing kernels (constant after construction)
  _wavelets    = null  // Morlet wavelet kernels per band
  _dftKernels  = null  // Hann-weighted DFT twiddle factors for delta bins

  // PPG / heart rate
  _ppgBuffer      = []           // rolling buffer of bandpass-filtered samples
  _ppgStepCount   = 0            // samples since last MSPTD run
  _hpPrevX        = 0            // high-pass: previous raw input
  _hpPrevY        = 0            // high-pass: previous output
  _lpPrevY        = 0            // low-pass: previous output

  // Heart-pulse oscillator
  _heartPhase    = 0
  _lastFrameTime = null

  // Accelerometer (head pose)
  _accSmooth = { x: 0, y: 0, z: 1 }  // starts pointing down (gravity reference)

  // Signal quality update counter
  _sqSampleCount = 0

  // Spectrogram — rolling columns of log₁₀(power) for display
  _specDisplay = []

  // Hi-res low-frequency spectrogram (0.5–8.0 Hz, 0.1 Hz resolution)
  _chBuffersLong  = [[], [], [], []]   // 2560-sample rolling buffers per channel
  _dftLoKernels   = null               // Hann-weighted twiddle factors for lo-freq bins
  _specLoDisplay  = []
  spectrumLoSampleCount = 0

  // ── Constructor ───────────────────────────────────────────────────────────

  constructor() {
    this._precomputeKernels()
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /** Filtered PPG waveform for display — same rolling buffer used by MSPTD. */
  get ppgDisplay() { return this._ppgBuffer }

  /** Rolling spectrogram columns (Float32Array[SPEC_BINS] each, log₁₀ power). */
  get spectrumDisplay() { return this._specDisplay }

  /** Hi-res low-freq spectrogram columns (Float32Array[SPEC_LO_NUM_BINS], log₁₀ power, 0.5–8 Hz @ 0.1 Hz). */
  get spectrumLoDisplay() { return this._specLoDisplay }

  /**
   * Connect to a Muse headset via Web Bluetooth and start streaming.
   * Subscribes to EEG, PPG (infrared), and accelerometer data.
   * Requires a user gesture and HTTPS (Web Bluetooth requirement).
   * @returns {Promise<void>}
   */
  async connect() {
    this._client = new MuseClient()
    this._client.enablePpg = true   // must set BEFORE connect()

    await this._client.connect()
    await this._client.start()
    this.isConnected = true
    this.deviceName  = this._client.deviceName ?? null

    // Device info (hw/fw/etc) for recording metadata — best-effort, non-fatal.
    try { this.deviceInfo = await this._client.deviceInfo() } catch { this.deviceInfo = null }

    this._resetState()

    // Raw per-electrode EEG packets — recorded verbatim in native muse-js shape
    // so files stay compatible with the eeg-recorder analysis pipeline. The
    // analysis (band powers, spectrogram) runs off the zipped stream below.
    this._eegRawSub = this._client.eegReadings.subscribe((reading) => {
      App.recordingManager?.recordEeg(reading)
    })

    // EEG bands — zipped per-sample stream feeds the signal-processing pipeline.
    this._eegSub = zipSamples(this._client.eegReadings).subscribe((sample) => {
      this._processEEGSample(sample.data)
    })

    // PPG — record every channel's raw packet; process infrared for heart rate.
    this._ppgSub = this._client.ppgReadings.subscribe((reading) => {
      App.recordingManager?.recordPpg(reading)
      if (reading.ppgChannel === PPG_INFRARED) {
        for (const s of reading.samples) this._processPPGSample(s)
      }
    })

    // IMU — accelerometer for head-tilt estimation
    this._accelSub = this._client.accelerometerData.subscribe((accel) => {
      App.recordingManager?.recordAccel(accel)
      this._processAccel(accel.samples)
    })

    // IMU — gyroscope
    this._gyroSub = this._client.gyroscopeData.subscribe((gyro) => {
      App.recordingManager?.recordGyro(gyro)
      this._processGyro(gyro.samples)
    })

    // Battery level from telemetry (updates every ~10 s)
    this._telemetrySub = this._client.telemetryData.subscribe((t) => {
      this.batteryLevel = Math.round(t.batteryLevel)
      this.onBatteryLevel?.(this.batteryLevel)
    })

    // Detect hardware-initiated disconnects
    this._client.connectionStatus.subscribe((connected) => {
      if (!connected && this.isConnected) this._handleDisconnect()
    })
  }

  /**
   * Gracefully disconnect from the Muse headset and reset all state.
   * Fires the onDisconnected callback if set.
   */
  disconnect() {
    this._eegSub?.unsubscribe()
    this._eegRawSub?.unsubscribe()
    this._ppgSub?.unsubscribe()
    this._accelSub?.unsubscribe()
    this._gyroSub?.unsubscribe()
    this._telemetrySub?.unsubscribe()
    this._client?.disconnect()
    this._handleDisconnect()
  }

  /**
   * Must be called every animation frame from App.update().
   * Advances the heart-pulse oscillator based on elapsed wall-clock time.
   * @param {number} now — performance.now() timestamp (ms)
   */
  update(now) {
    if (this._lastFrameTime === null) {
      this._lastFrameTime = now
    }
    const dt = Math.min((now - this._lastFrameTime) / 1000, 0.1)  // clamp to 100 ms
    this._lastFrameTime = now

    if (this.isConnected) {
      // Advance phase at current heartRate
      this._heartPhase += (this.heartRate / 60) * 2 * Math.PI * dt
      // Shape into a cardiac-like spike: sharp systolic peak, slow diastolic decay
      const s = (Math.sin(this._heartPhase) + 1) / 2
      this.heartPulse = s * s * s              // cube sharpens the peak
    } else {
      this.heartPulse = 0
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  _resetState() {
    this._chBuffers     = [[], [], [], []]
    this._analysisCount = 0
    this._apModel       = { a: 0, b: -1.5 }
    this._apWindowCount = 0
    this._apRefitCount  = 0
    this._ppgBuffer    = []
    this._ppgStepCount = 0
    this._hpPrevX = this._hpPrevY = this._lpPrevY = 0
    this._sqSampleCount = 0
    this.bandPower    = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
    this.heartRate    = 70
    this.heartPulse   = 0
    this.headPose     = { pitch: 0, roll: 0 }
    this.eegChannels   = [[], [], [], []]
    this.accelDisplay  = { x: [], y: [], z: [] }
    this.gyroDisplay   = { x: [], y: [], z: [] }
    this.signalQuality = ['poor', 'poor', 'poor', 'poor']
    this.eegSampleCount      = 0
    this.ppgSampleCount      = 0
    this.imuSampleCount      = 0
    this.spectrumSampleCount = 0
    this._specDisplay = []
    this._chBuffersLong = [[], [], [], []]
    this._specLoDisplay = []
    this.spectrumLoSampleCount = 0
  }

  _handleDisconnect() {
    this.isConnected = false
    this.batteryLevel = null
    this.deviceName = null
    this.deviceInfo = null
    this._resetState()
    this.onDisconnected?.()
  }

  // ── EEG spectral band powers ──────────────────────────────────────────────

  _processEEGSample(channelData) {
    this.eegSampleCount++

    for (let ch = 0; ch < 4; ch++) {
      const v = channelData[ch]
      const val = isNaN(v) ? 0 : v

      if (!isNaN(v)) {
        this.eegChannels[ch].push(v)
        if (this.eegChannels[ch].length > EEG_DISPLAY_LEN) this.eegChannels[ch].shift()
      }

      // Rolling analysis buffer — push new sample, drop oldest if full
      this._chBuffers[ch].push(val)
      if (this._chBuffers[ch].length > EEG_BUF_SIZE) this._chBuffers[ch].shift()

      // Long buffer for hi-res low-frequency spectrogram (10 s window)
      this._chBuffersLong[ch].push(val)
      if (this._chBuffersLong[ch].length > SPEC_LO_BUF_SIZE) this._chBuffersLong[ch].shift()
    }

    // Update signal quality ~4× per second (every 64 samples at 256 Hz)
    this._sqSampleCount++
    if (this._sqSampleCount >= 64) {
      this._sqSampleCount = 0
      this._updateSignalQuality()
    }

    // Trigger spectral analysis every EEG_BUF_SIZE/2 new samples (50% overlap, ~2×/s)
    this._analysisCount++
    if (this._analysisCount >= EEG_BUF_SIZE / 2) {
      this._analysisCount = 0
      if (this._chBuffers[0].length >= EEG_BUF_SIZE) {
        this._computeBandPower()
      }
    }
  }

  _updateSignalQuality() {
    for (let ch = 0; ch < 4; ch++) {
      const buf = this.eegChannels[ch]
      const n = Math.min(buf.length, SQ_WIN)
      if (n < 10) { this.signalQuality[ch] = 'poor'; continue }
      const slice = buf.slice(-n)
      const mean = slice.reduce((a, b) => a + b, 0) / n
      const rms = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
      if (rms < SQ_LOW)       this.signalQuality[ch] = 'good'
      else if (rms < SQ_HIGH) this.signalQuality[ch] = 'marginal'
      else                    this.signalQuality[ch] = 'poor'
    }
  }

  /**
   * Compute per-channel band powers, apply quality-weighted aggregation, then
   * normalise against the fitted aperiodic (1/f) background.
   *
   * Pipeline:
   *   1. Per channel: sparse Hann-DFT for delta (1–4 Hz) + Morlet wavelet power
   *      for theta/alpha/beta/gamma.
   *   2. Quality-weighted channel aggregation (up to 2 bad channels dropped).
   *   3. Periodic log-log linear fit to the wavelet powers → aperiodic model.
   *      First fit uses instant adoption; subsequent fits use EMA smoothing.
   *   4. Divide each band by its expected aperiodic power, then renormalise to
   *      sum=1. Genuine oscillatory elevations (e.g. alpha peak during eyes-closed)
   *      get a proportionally larger share after 1/f correction.
   */
  _computeBandPower() {
    // Step 1: per-channel band power
    const chBands = this._chBuffers.map(sig => this._computeChannelBands(sig))

    // Step 2: quality-weighted aggregation
    const weights = this._getChannelWeights()
    const totalW  = weights.reduce((a, b) => a + b, 0)

    const raw = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
    for (const band of Object.keys(raw)) {
      let sum = 0
      for (let ch = 0; ch < 4; ch++) sum += chBands[ch][band] * weights[ch]
      raw[band] = totalW > 0 ? sum / totalW : 0
    }

    // Step 3: periodic aperiodic model refit
    this._apWindowCount++
    if (this._apWindowCount >= APERIODIC_UPDATE_INTERVAL) {
      this._apWindowCount = 0
      this._refitAperiodicModel(chBands, weights, totalW)
    }

    // Step 4: aperiodic normalisation → relative power (sum=1)
    const result = this._apNormalize(raw)

    // Step 4b: EMA smooth to avoid staircase jumps between ~2 Hz analysis windows
    for (const band of Object.keys(result)) {
      this.bandPower[band] += BAND_SMOOTH * (result[band] - this.bandPower[band])
    }

    App.recordingManager?.recordBands(this.bandPower)

    // Step 5: spectrogram columns (uses same channel weights)
    this._computeSpectrum(weights, totalW)
    this._computeSpectrumLo(weights, totalW)

    this._debugLog(raw, result, weights)
  }

  /**
   * Compute band powers for a single EEG channel.
   *   delta  — Hann-windowed DFT sum at bins 1, 2, 3 Hz (precomputed twiddle factors).
   *   theta/alpha/beta/gamma — mean Morlet wavelet power over the valid central window.
   *
   * @param {number[]} sig — EEG_BUF_SIZE samples
   * @returns {{ delta, theta, alpha, beta, gamma }}
   */
  _computeChannelBands(sig) {
    // Delta: sparse Hann-DFT
    let delta = 0
    for (const k of DFT_BINS) {
      const { re: reK, im: imK } = this._dftKernels[k]
      let r = 0, m = 0
      for (let n = 0; n < EEG_BUF_SIZE; n++) {
        r += reK[n] * sig[n]
        m += imK[n] * sig[n]
      }
      delta += r * r + m * m
    }

    return {
      delta,
      theta: this._waveletPower(sig, 'theta'),
      alpha: this._waveletPower(sig, 'alpha'),
      beta:  this._waveletPower(sig, 'beta'),
      gamma: this._waveletPower(sig, 'gamma'),
    }
  }

  /**
   * Mean Morlet wavelet power over the valid (edge-free) portion of a signal window.
   * Uses the precomputed kernel for the given band (see _precomputeKernels).
   *
   * Only samples [halfWin … N-1-halfWin] are used, so the kernel is always
   * fully contained within the buffer (no zero-padding artefacts).
   *
   * @param {number[]} sig  — EEG_BUF_SIZE samples
   * @param {string}   band — key of WAVELET_BAND_FREQS
   * @returns {number} mean |W(t)|² over valid samples
   */
  _waveletPower(sig, band) {
    const { re, im, halfWin } = this._wavelets[band]
    const N    = sig.length
    const kLen = re.length
    const start = halfWin
    const end   = N - 1 - halfWin

    if (start > end) return 0

    let totalPower = 0
    let count = 0
    for (let i = start; i <= end; i++) {
      let r = 0, m = 0
      const base = i - halfWin
      for (let k = 0; k < kLen; k++) {
        const s = sig[base + k]
        r += re[k] * s
        m += im[k] * s
      }
      totalPower += r * r + m * m
      count++
    }
    return count > 0 ? totalPower / count : 0
  }

  /**
   * Compute normalised channel weights for the quality-weighted average.
   *
   * Algorithm:
   *   1. Assign a quality score: good=2, marginal=1, poor=0.
   *   2. Channels at or below the drop threshold (controlled by badChannelThreshold)
   *      are candidates for exclusion.
   *   3. Sort candidates worst-first; drop at most 2 (always retain ≥ 2 channels).
   *   4. Assign quality weights to retained channels (good=1, marginal=marginalChannelWeight,
   *      poor=0) and normalise to sum=1.
   *   5. If all retained channels have weight 0 (all poor, threshold='poor'), fall back to
   *      equal weights across non-dropped channels.
   *
   * @returns {number[]} length-4 weight array, sums to 1
   */
  _getChannelWeights() {
    const QUALITY_SCORE  = { good: 2, marginal: 1, poor: 0 }
    const QUALITY_WEIGHT = { good: 1.0, marginal: this.marginalChannelWeight, poor: 0.0 }

    // Channels eligible to be dropped, sorted worst-first
    const dropScore = this.badChannelThreshold === 'marginal' ? 1 : 0
    const candidates = [0, 1, 2, 3]
      .filter(ch => QUALITY_SCORE[this.signalQuality[ch]] <= dropScore)
      .sort((a, b) => QUALITY_SCORE[this.signalQuality[a]] - QUALITY_SCORE[this.signalQuality[b]])

    const dropped = new Set(candidates.slice(0, 2))   // drop at most 2

    const weights = this.signalQuality.map((q, ch) => dropped.has(ch) ? 0 : QUALITY_WEIGHT[q])
    const total   = weights.reduce((a, b) => a + b, 0)

    if (total > 0) return weights.map(w => w / total)

    // Fallback: equal weight to all non-dropped channels
    const fallback  = [0, 0, 0, 0]
    const active    = [0, 1, 2, 3].filter(ch => !dropped.has(ch))
    for (const ch of active) fallback[ch] = 1 / active.length
    return fallback
  }

  /**
   * Refit the aperiodic background model using quality-weighted wavelet powers.
   *
   * Follows BOSC_bgfit: linear regression of log₁₀(power) vs log₁₀(frequency)
   * at the band representative frequencies (6, 10, 20, 40 Hz).
   * Model parameters are updated via exponential moving average (AP_SMOOTH)
   * to avoid abrupt jumps.
   *
   * @param {Array}  chBands — per-channel { delta, theta, alpha, beta, gamma }
   * @param {number[]} weights — normalised channel weights (from _getChannelWeights)
   * @param {number}   totalW  — sum of weights (may be < 1 before normalisation)
   */
  _refitAperiodicModel(chBands, weights, totalW) {
    if (totalW === 0) return

    const logF = AP_FIT_FREQS.map(f => Math.log10(f))
    const logP = AP_FIT_BANDS.map(band => {
      let sum = 0
      for (let ch = 0; ch < 4; ch++) sum += chBands[ch][band] * weights[ch]
      const avgP = sum / totalW
      return avgP > 0 ? Math.log10(avgP) : -10
    })

    const { a, b } = this._linReg(logF, logP)

    // First fit adopts the regression instantly (no blending with the dummy prior);
    // subsequent fits use EMA smoothing to avoid abrupt jumps.
    const smooth = this._apRefitCount === 0 ? 1.0 : AP_SMOOTH
    this._apModel.a = (1 - smooth) * this._apModel.a + smooth * a
    this._apModel.b = (1 - smooth) * this._apModel.b + smooth * b

    this._apRefitCount++
  }

  /**
   * Normalise raw band powers against the aperiodic (1/f) background model,
   * then return relative power values that sum to 1.
   *
   * Each band's power is divided by the expected power at its representative
   * frequency under the fitted log-log linear model:
   *   expected(f) = 10^(a + b·log₁₀(f))
   *
   * This means a band with power exactly on the aperiodic baseline contributes
   * its "fair share" to the total, while a genuinely elevated oscillation
   * (e.g. strong alpha during eyes-closed) receives a proportionally larger
   * share — the 1/f correction prevents low-frequency bands from dominating
   * simply because 1/f power is higher there.
   *
   * During warm-up (< AP_MIN_REFITS model refits, ≈ 15 s), returns zeros
   * so no EEG influence reaches the shaders before the model has converged.
   *
   * @param {{ delta, theta, alpha, beta, gamma }} raw — quality-weighted band powers
   * @returns {{ delta, theta, alpha, beta, gamma }} sum=1 relative power (or all zeros during warm-up)
   */
  _apNormalize(raw) {
    if (this._apRefitCount < AP_MIN_REFITS) {
      return { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
    }

    const { a, b } = this._apModel
    const BAND_FREQ = { delta: 2, theta: 6, alpha: 10, beta: 20, gamma: 40 }

    // Only active (mapped) bands participate in the relative-power sum.
    // Inactive bands are zeroed so they cannot inflate or dominate the total.
    let total = 0
    const norm = {}
    for (const [band, f] of Object.entries(BAND_FREQ)) {
      if (!this.normalizeBands.has(band)) {
        norm[band] = 0
        continue
      }
      const expected = Math.pow(10, a + b * Math.log10(f))
      norm[band] = raw[band] > 0 ? raw[band] / expected : 0
      total += norm[band]
    }

    if (total === 0) {
      // Equal-weight fallback only for active bands
      const n = this.normalizeBands.size
      const eq = n > 0 ? 1 / n : 0
      for (const band of Object.keys(BAND_FREQ)) {
        norm[band] = this.normalizeBands.has(band) ? eq : 0
      }
      return norm
    }

    for (const band of Object.keys(norm)) {
      if (this.normalizeBands.has(band)) norm[band] /= total
    }
    return norm
  }

  /**
   * Log pipeline internals when this.debug is true.
   * Enable at runtime: App.eegManager.debug = true
   */
  _debugLog(raw, result, weights) {
    if (!this.debug) return
    const { a, b } = this._apModel
    const BAND_FREQ = { delta: 2, theta: 6, alpha: 10, beta: 20, gamma: 40 }
    const expected = {}
    for (const [band, f] of Object.entries(BAND_FREQ)) {
      expected[band] = Math.pow(10, a + b * Math.log10(f))
    }
    const fmt = (obj) => {
      const o = {}
      for (const [k, v] of Object.entries(obj)) o[k] = +v.toFixed(4)
      return o
    }
    console.groupCollapsed('[EEG] band power pipeline')
    console.log('signal quality:', [...this.signalQuality])
    console.log('channel weights:', weights.map(w => +w.toFixed(3)))
    console.log('raw wavelet power:', fmt(raw))
    console.log(`aperiodic model: a=${a.toFixed(4)}, b=${b.toFixed(4)} (refits: ${this._apRefitCount})`)
    console.log('expected (1/f):', fmt(expected))
    console.log('output (sum=1):', fmt(result))
    console.groupEnd()
  }

  /**
   * Compute a quality-weighted Hann-DFT spectrogram column (bins 1–50 Hz).
   * Stores log₁₀(power) in a rolling buffer for BioDataDisplay rendering.
   * Artifact-robust auto-scaling is handled in BioDataDisplay (percentile window + cap).
   *
   * @param {number[]} weights — normalised channel weights
   * @param {number}   totalW  — sum of weights
   */
  _computeSpectrum(weights, totalW) {
    if (totalW === 0) return

    const spectrum = new Float32Array(SPEC_BINS)
    for (let k = 1; k <= SPEC_BINS; k++) {
      const { re: reK, im: imK } = this._dftKernels[k]
      let power = 0
      for (let ch = 0; ch < 4; ch++) {
        if (weights[ch] === 0) continue
        const sig = this._chBuffers[ch]
        let r = 0, m = 0
        for (let n = 0; n < EEG_BUF_SIZE; n++) {
          r += reK[n] * sig[n]
          m += imK[n] * sig[n]
        }
        power += (r * r + m * m) * weights[ch]
      }
      spectrum[k - 1] = Math.log10(power / totalW + 1e-10)
    }

    this._specDisplay.push(spectrum)
    if (this._specDisplay.length > SPEC_COLS) this._specDisplay.shift()
    this.spectrumSampleCount++
  }

  /**
   * Hi-res low-frequency spectrogram (0.5–8.0 Hz at 0.1 Hz resolution).
   * Uses the 2560-sample (10 s) long buffer for true sub-Hz frequency resolution.
   * Artifact-robust auto-scaling is handled in BioDataDisplay (percentile window + cap).
   */
  _computeSpectrumLo(weights, totalW) {
    if (totalW === 0) return
    const N = SPEC_LO_BUF_SIZE
    if (this._chBuffersLong[0].length < N) return   // need full 10 s window

    const spectrum = new Float32Array(SPEC_LO_NUM_BINS)
    for (let k = 0; k < SPEC_LO_NUM_BINS; k++) {
      const { re: reK, im: imK } = this._dftLoKernels[k]
      let power = 0
      for (let ch = 0; ch < 4; ch++) {
        if (weights[ch] === 0) continue
        const sig = this._chBuffersLong[ch]
        let r = 0, m = 0
        for (let n = 0; n < N; n++) {
          r += reK[n] * sig[n]
          m += imK[n] * sig[n]
        }
        power += (r * r + m * m) * weights[ch]
      }
      spectrum[k] = Math.log10(power / totalW + 1e-10)
    }

    this._specLoDisplay.push(spectrum)
    if (this._specLoDisplay.length > SPEC_COLS) this._specLoDisplay.shift()
    this.spectrumLoSampleCount++
  }

  /**
   * Ordinary-least-squares linear regression: y = a + b·x.
   * @param {number[]} xs
   * @param {number[]} ys
   * @returns {{ a: number, b: number }}
   */
  _linReg(xs, ys) {
    const n = xs.length
    let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0
    for (let i = 0; i < n; i++) {
      sumX += xs[i]; sumY += ys[i]; sumXX += xs[i] * xs[i]; sumXY += xs[i] * ys[i]
    }
    const denom = n * sumXX - sumX * sumX
    if (Math.abs(denom) < 1e-12) return { a: sumY / n, b: 0 }
    const b = (n * sumXY - sumX * sumY) / denom
    const a = (sumY - b * sumX) / n
    return { a, b }
  }

  /**
   * Precompute all signal-processing kernels once at construction time.
   *
   * Morlet wavelet kernels (following BOSC_tf.m):
   *   σ = τ / (2π·f)                     — temporal std dev in seconds
   *   A = 1 / sqrt(σ·√π)                 — amplitude normalisation
   *   window = ±3σ samples               — ±3.6σ used in BOSC_tf, reduced to ±3σ
   *                                         so the 6 Hz kernel fits in 256 samples
   *   kernel[i] = A·exp(-t²/2σ²)·exp(i2πft)   (real and imaginary stored separately)
   *
   * DFT twiddle factors (Hann-weighted) for delta-band bins:
   *   twiddle_re[k][n] = hann(n) · cos(2π·k·n / N)
   *   twiddle_im[k][n] = hann(n) · sin(2π·k·n / N)
   *   Precomputing these eliminates repeated cos/sin calls in the hot path.
   */
  _precomputeKernels() {
    const N = EEG_BUF_SIZE

    // ── Morlet wavelets ────────────────────────────────────────────────────
    this._wavelets = {}
    for (const [band, f] of Object.entries(WAVELET_BAND_FREQS)) {
      const tau     = band === 'theta' ? WAVELET_WAVENUMBER_THETA : WAVELET_WAVENUMBER
      const sigma   = tau / (2 * Math.PI * f)                         // seconds
      const A       = 1 / Math.sqrt(sigma * Math.sqrt(Math.PI))        // BOSC amplitude norm
      const halfWin = Math.ceil(3.0 * sigma * EEG_FS)                  // ±3σ in samples
      const len     = 2 * halfWin + 1
      const re      = new Float32Array(len)
      const im      = new Float32Array(len)

      for (let i = 0; i < len; i++) {
        const t     = (i - halfWin) / EEG_FS
        const gauss = A * Math.exp(-(t * t) / (2 * sigma * sigma))
        re[i] = gauss * Math.cos(2 * Math.PI * f * t)
        im[i] = gauss * Math.sin(2 * Math.PI * f * t)
      }

      this._wavelets[band] = { re, im, halfWin }
    }

    // ── Hann-weighted DFT twiddle factors (bins 1–SPEC_BINS) ──────────────
    // Bins 1–3 used for delta band power; full range 1–SPEC_BINS used for spectrogram.
    this._dftKernels = {}
    const hann = new Float32Array(N)
    for (let n = 0; n < N; n++) {
      hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1))
    }
    for (let k = 1; k <= SPEC_BINS; k++) {
      const reK = new Float32Array(N)
      const imK = new Float32Array(N)
      for (let n = 0; n < N; n++) {
        const angle = (2 * Math.PI * k * n) / N
        reK[n] = hann[n] * Math.cos(angle)
        imK[n] = hann[n] * Math.sin(angle)
      }
      this._dftKernels[k] = { re: reK, im: imK }
    }

    // ── Hi-res lo-freq DFT twiddle factors (0.5–8.0 Hz, 0.1 Hz steps, N=2560) ─
    const NLo = SPEC_LO_BUF_SIZE
    const hannLo = new Float32Array(NLo)
    for (let n = 0; n < NLo; n++) {
      hannLo[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (NLo - 1))
    }
    this._dftLoKernels = new Array(SPEC_LO_NUM_BINS)
    for (let k = 0; k < SPEC_LO_NUM_BINS; k++) {
      const fHz = SPEC_LO_MIN_HZ + k * SPEC_LO_STEP_HZ
      const reK = new Float32Array(NLo)
      const imK = new Float32Array(NLo)
      for (let n = 0; n < NLo; n++) {
        const angle = (2 * Math.PI * fHz * n) / EEG_FS
        reK[n] = hannLo[n] * Math.cos(angle)
        imK[n] = hannLo[n] * Math.sin(angle)
      }
      this._dftLoKernels[k] = { re: reK, im: imK }
    }
  }

  // ── PPG / heart rate ──────────────────────────────────────────────────────

  /**
   * Process one raw PPG sample through the filter pipeline, then trigger
   * MSPTDfast v2 batch detection every PPG_RUN_STEP samples.
   *
   * Pipeline:
   *   1. First-order IIR high-pass  @ 0.5 Hz — removes DC & baseline wander
   *   2. First-order IIR low-pass   @ 3.5 Hz — removes motion artifacts
   *   3. Accumulate into rolling 6-second buffer
   *   4. Every ~1 second, run MSPTDfast on the buffer to update heartRate
   */
  _processPPGSample(raw) {
    // 1. High-pass (remove DC & baseline wander)
    const hp = HP_ALPHA * (this._hpPrevY + raw - this._hpPrevX)
    this._hpPrevX = raw
    this._hpPrevY = hp

    // 2. Low-pass (smooth noise)
    const lp = (1 - LP_ALPHA) * this._lpPrevY + LP_ALPHA * hp
    this._lpPrevY = lp

    // Rolling 6-second window
    this.ppgSampleCount++
    this._ppgBuffer.push(lp)
    if (this._ppgBuffer.length > PPG_WIN_SAMP) this._ppgBuffer.shift()

    // Run MSPTDfast every PPG_RUN_STEP new samples once the window is full
    this._ppgStepCount++
    if (this._ppgStepCount >= PPG_RUN_STEP) {
      this._ppgStepCount = 0
      this._runMSPTD()
    }
  }

  /**
   * Run MSPTDfast v2 on the current 6-second PPG buffer.
   * Downsamples to ~20 Hz, detects peaks, corrects back to original fs,
   * then computes heart rate from the median inter-beat interval.
   */
  _runMSPTD() {
    if (this._ppgBuffer.length < PPG_WIN_SAMP) return

    const win = this._ppgBuffer.slice(-PPG_WIN_SAMP)

    // Downsample: simple decimation (bandpass already limits aliasing)
    const ds = []
    for (let i = 0; i < win.length; i += DS_FACTOR) ds.push(win[i])

    // Detect peaks in downsampled signal
    const { peaks: dsPeaks } = _msptdDetect(ds, DS_FS)
    if (dsPeaks.length < 2) return

    // Upsample indices and refine to true local maximum in original-fs window
    const refined = dsPeaks.map(p => {
      const center = p * DS_FACTOR
      const lo = Math.max(0, center - REFINE_TOL)
      const hi = Math.min(win.length - 1, center + REFINE_TOL)
      let maxVal = -Infinity, maxIdx = center
      for (let i = lo; i <= hi; i++) {
        if (win[i] > maxVal) { maxVal = win[i]; maxIdx = i }
      }
      return maxIdx
    })

    // Sort and enforce refractory period (removes duplicates from refinement)
    refined.sort((a, b) => a - b)
    const peaks = [refined[0]]
    for (let i = 1; i < refined.length; i++) {
      if (refined[i] - peaks[peaks.length - 1] >= MIN_PEAK_DIST) peaks.push(refined[i])
    }

    if (peaks.length < 2) return

    // Compute IBIs from all consecutive peak pairs in this window
    const ibis = []
    for (let i = 1; i < peaks.length; i++) {
      const ibi = (peaks[i] - peaks[i - 1]) / PPG_FS
      const hr  = 60 / ibi
      if (hr >= HR_MIN && hr <= HR_MAX) ibis.push(ibi)
    }
    if (ibis.length === 0) return

    // Median IBI → stable heart rate estimate
    const sorted = [...ibis].sort((a, b) => a - b)
    const medIBI = sorted[Math.floor(sorted.length / 2)]
    this.heartRate = Math.round(60 / medIBI)

    App.recordingManager?.recordHr(this.heartRate)
  }

  // ── IMU / head pose ──────────────────────────────────────────────────────

  /**
   * Derive head tilt angles from the accelerometer gravity vector.
   * Applies exponential low-pass (α=0.08) to reject vibration/jerk.
   *
   * Angles (radians, relative to upright neutral position):
   *   pitch — forward/backward nod   (rotation around X)
   *   roll  — left/right side tilt   (rotation around Z)
   */
  _processAccel(samples) {
    // Average all samples in the reading (3 per packet at 52 Hz)
    let ax = 0, ay = 0, az = 0
    for (const s of samples) { ax += s.x; ay += s.y; az += s.z }
    ax /= samples.length
    ay /= samples.length
    az /= samples.length

    // Push raw average to display buffer
    this.imuSampleCount++
    this.accelDisplay.x.push(ax)
    this.accelDisplay.y.push(ay)
    this.accelDisplay.z.push(az)
    if (this.accelDisplay.x.length > IMU_DISPLAY_LEN) {
      this.accelDisplay.x.shift()
      this.accelDisplay.y.shift()
      this.accelDisplay.z.shift()
    }

    // Exponential moving average — only slow head movements pass through
    this._accSmooth.x = (1 - ACC_ALPHA) * this._accSmooth.x + ACC_ALPHA * ax
    this._accSmooth.y = (1 - ACC_ALPHA) * this._accSmooth.y + ACC_ALPHA * ay
    this._accSmooth.z = (1 - ACC_ALPHA) * this._accSmooth.z + ACC_ALPHA * az

    const { x, y, z } = this._accSmooth
    this.headPose = {
      pitch: Math.atan2(-x, Math.sqrt(y * y + z * z)),
      roll:  Math.atan2(y, z),
    }
  }

  _processGyro(samples) {
    let gx = 0, gy = 0, gz = 0
    for (const s of samples) { gx += s.x; gy += s.y; gz += s.z }
    gx /= samples.length
    gy /= samples.length
    gz /= samples.length

    this.gyroDisplay.x.push(gx)
    this.gyroDisplay.y.push(gy)
    this.gyroDisplay.z.push(gz)
    if (this.gyroDisplay.x.length > IMU_DISPLAY_LEN) {
      this.gyroDisplay.x.shift()
      this.gyroDisplay.y.shift()
      this.gyroDisplay.z.shift()
    }
  }
}
