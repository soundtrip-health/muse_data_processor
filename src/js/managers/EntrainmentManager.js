import App from '../App'

// ── Constants ─────────────────────────────────────────────────────────────────

// Audio novelty resampling
const NOVELTY_FS      = 64          // Hz — uniform grid for resampled audio novelty
const TEMPO_WIN_SEC   = 8.0         // seconds — tempo analysis window (Stober §4.1)
const TEMPO_WIN_SAMP  = Math.round(NOVELTY_FS * TEMPO_WIN_SEC)  // 512

// DFT frequency grid for tempograms (0.5–5.0 Hz = 30–300 BPM)
const RHYTHM_MIN_HZ   = 0.5
const RHYTHM_MAX_HZ   = 5.0
const RHYTHM_STEP_HZ  = 0.1
const RHYTHM_NUM_BINS  = Math.round((RHYTHM_MAX_HZ - RHYTHM_MIN_HZ) / RHYTHM_STEP_HZ) + 1 // 46

// EEG novelty curve
const EEG_FS          = 256
const EEG_AVG_WIN     = 128         // 0.5 s moving-average window (Stober §3.2)
const EEG_TEMPO_BUF   = Math.round(EEG_FS * TEMPO_WIN_SEC)  // 2048

// Entrainment index
const PEAK_Z_THRESH   = 0.5         // audio z-score threshold for "beat frequency"
const SIGMOID_K       = 2.0         // sigmoid steepness
const ENTRAIN_EMA     = 0.15        // output smoothing (~3 s settling at 2 Hz)
const UPDATE_INTERVAL_MS = 500      // ~2 Hz update cadence

// Quality scoring (mirrors EEGManager conventions)
const Q_WEIGHT = { good: 1.0, marginal: 0.5, poor: 0.0 }

// ── EntrainmentManager ───────────────────────────────────────────────────────

/**
 * EntrainmentManager
 *
 * Computes a real-time neural entrainment index by comparing parallel
 * tempograms derived from the audio spectral-flux novelty curve and the
 * mean-subtracted EEG signal.
 *
 * References:
 *   - Nozaradan et al. (2012) — selective SS-EP enhancement at beat/meter frequencies
 *   - Stober et al. (2016) "Brain Beats" — tempogram extraction from EEG
 */
export default class EntrainmentManager {
  // ── Public output ──────────────────────────────────────────────────────────
  entrainment      = 0       // smoothed 0–1 index
  audioTempogram   = null    // Float32Array(46) | null — for display
  eegTempogram     = null    // Float32Array(46) | null — for display

  // ── Private ────────────────────────────────────────────────────────────────
  _audioKernels    = null    // precomputed Hann-weighted DFT kernels for audio novelty
  _eegKernels      = null    // precomputed Hann-weighted DFT kernels for EEG novelty
  _resampleBuf     = null    // Float32Array(TEMPO_WIN_SAMP)
  _eegNoveltyBuf   = null    // Float32Array(EEG_TEMPO_BUF)
  _lastUpdateTime  = 0

  constructor() {
    this._resampleBuf   = new Float32Array(TEMPO_WIN_SAMP)
    this._eegNoveltyBuf = new Float32Array(EEG_TEMPO_BUF)
    this._precomputeKernels()
  }

  // ── Kernel precomputation ──────────────────────────────────────────────────

  _precomputeKernels() {
    this._audioKernels = _buildHannDFTKernels(TEMPO_WIN_SAMP, NOVELTY_FS)
    this._eegKernels   = _buildHannDFTKernels(EEG_TEMPO_BUF, EEG_FS)
  }

  // ── Audio tempogram ────────────────────────────────────────────────────────

  /**
   * Linear-interpolate the variable-rate novelty ring buffer onto a uniform
   * NOVELTY_FS grid covering the most recent TEMPO_WIN_SEC seconds.
   * Returns false if insufficient data (< 4 s).
   */
  _resampleNovelty(ring) {
    if (ring.count < 2) return false

    // Extract (time, value) pairs in chronological order from ring buffer
    const n     = ring.count
    const cap   = ring.times.length
    const start = (ring.head - n + cap) % cap

    // Find time bounds
    const tNewest = ring.times[(ring.head - 1 + cap) % cap]
    const tOldest = ring.times[start]
    const span    = tNewest - tOldest
    if (span < 4.0) return false   // need at least 4 s of data

    const tStart = tNewest - TEMPO_WIN_SEC
    const dt     = 1 / NOVELTY_FS
    const buf    = this._resampleBuf

    // Walk through ring buffer for interpolation
    // Build a sorted view of the most recent samples
    let ri = 0  // ring read index (into chronological order)
    for (let i = 0; i < TEMPO_WIN_SAMP; i++) {
      const t = tStart + i * dt

      // Advance ring index until we bracket t
      while (ri < n - 1) {
        const nextIdx = (start + ri + 1) % cap
        if (ring.times[nextIdx] >= t) break
        ri++
      }

      const idxA = (start + ri) % cap
      const idxB = (start + Math.min(ri + 1, n - 1)) % cap
      const tA   = ring.times[idxA]
      const tB   = ring.times[idxB]

      if (tB === tA || ri >= n - 1) {
        buf[i] = ring.values[idxA]
      } else {
        const frac = (t - tA) / (tB - tA)
        buf[i] = ring.values[idxA] + frac * (ring.values[idxB] - ring.values[idxA])
      }
    }

    return true
  }

  /** DFT dot products on the resampled audio novelty buffer. */
  _computeAudioTempogram() {
    return _dftPower(this._resampleBuf, this._audioKernels)
  }

  // ── EEG tempogram ──────────────────────────────────────────────────────────

  /**
   * Build a quality-weighted, mean-subtracted EEG novelty curve from the
   * most recent EEG_TEMPO_BUF samples of the long channel buffers.
   * Returns false if insufficient data.
   */
  _computeEEGNovelty(eegMgr) {
    const chBufs = eegMgr._chBuffersLong
    if (!chBufs || chBufs[0].length < EEG_TEMPO_BUF) return false

    // Quality-weighted channel aggregation
    const sq = eegMgr.signalQuality
    const weights = sq.map(q => Q_WEIGHT[q] ?? 0)
    const totalW  = weights.reduce((a, b) => a + b, 0)
    if (totalW === 0) return false

    const N   = EEG_TEMPO_BUF
    const buf = this._eegNoveltyBuf

    // Weighted average of last N samples across channels
    for (let i = 0; i < N; i++) {
      let sum = 0
      for (let ch = 0; ch < 4; ch++) {
        if (weights[ch] === 0) continue
        const sig = chBufs[ch]
        sum += sig[sig.length - N + i] * weights[ch]
      }
      buf[i] = sum / totalW
    }

    // Subtract 0.5 s (EEG_AVG_WIN samples) centered moving average (Stober §3.2).
    // Prefix sum → O(N) computation of local means, then subtract in-place.
    const half   = EEG_AVG_WIN >> 1
    const prefix = new Float32Array(N + 1)
    for (let i = 0; i < N; i++) prefix[i + 1] = prefix[i] + buf[i]
    for (let i = 0; i < N; i++) {
      const lo   = Math.max(0, i - half)
      const hi   = Math.min(N - 1, i + half)
      const mean = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1)
      buf[i] -= mean
    }

    return true
  }

  /** DFT dot products on the EEG novelty buffer. */
  _computeEEGTempogram() {
    return _dftPower(this._eegNoveltyBuf, this._eegKernels)
  }

  // ── Entrainment index ──────────────────────────────────────────────────────

  /**
   * Compare audio and EEG tempograms using Nozaradan-inspired z-score
   * selective enhancement. Returns raw entrainment value in [0, 1].
   */
  _computeEntrainment(audioSpec, eegSpec) {
    // Z-score normalize both spectra
    const audioZ = _zscore(audioSpec)
    const eegZ   = _zscore(eegSpec)
    if (!audioZ || !eegZ) return 0

    // Identify beat peaks in audio tempogram
    let peakSum = 0, peakCount = 0
    let nonSum  = 0, nonCount  = 0

    for (let i = 0; i < RHYTHM_NUM_BINS; i++) {
      if (audioZ[i] > PEAK_Z_THRESH) {
        peakSum += eegZ[i]
        peakCount++
      } else {
        nonSum += eegZ[i]
        nonCount++
      }
    }

    if (peakCount === 0 || nonCount === 0) return 0

    // Contrast: how much more EEG power at beat frequencies vs non-beat
    const contrast = (peakSum / peakCount) - (nonSum / nonCount)

    // Sigmoid → [0, 1], then rescale so 0.5 → 0
    const sigmoid = 1 / (1 + Math.exp(-SIGMOID_K * contrast))
    return Math.max(0, 2 * (sigmoid - 0.5))
  }

  // ── Per-frame update ───────────────────────────────────────────────────────

  update(now) {
    if (now - this._lastUpdateTime < UPDATE_INTERVAL_MS) return
    this._lastUpdateTime = now

    const audioMgr = App.audioManager
    const eegMgr   = App.eegManager

    // Audio tempogram (works without EEG)
    if (audioMgr?.isPlaying && audioMgr.noveltyRing.count > 0) {
      if (this._resampleNovelty(audioMgr.noveltyRing)) {
        this.audioTempogram = this._computeAudioTempogram()
      }
    } else {
      this.audioTempogram = null
    }

    // EEG tempogram (works without audio)
    if (eegMgr?.isConnected) {
      if (this._computeEEGNovelty(eegMgr)) {
        this.eegTempogram = this._computeEEGTempogram()
      }
    } else {
      this.eegTempogram = null
    }

    // Entrainment (requires both)
    if (this.audioTempogram && this.eegTempogram) {
      const raw = this._computeEntrainment(this.audioTempogram, this.eegTempogram)
      this.entrainment += ENTRAIN_EMA * (raw - this.entrainment)
    } else {
      this.entrainment *= (1 - ENTRAIN_EMA)
    }
  }
}

// ── Shared helpers ───────────────────────────────────────────────────────────

/**
 * Precompute Hann-weighted DFT twiddle factors for RHYTHM_NUM_BINS frequency
 * bins spanning RHYTHM_MIN_HZ–RHYTHM_MAX_HZ at RHYTHM_STEP_HZ resolution.
 *
 * @param {number} N    — window length in samples
 * @param {number} fs   — sample rate of the signal being analyzed
 * @returns {Array<{re: Float32Array, im: Float32Array}>}
 */
function _buildHannDFTKernels(N, fs) {
  const hann = new Float32Array(N)
  for (let n = 0; n < N; n++) {
    hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1))
  }

  const kernels = new Array(RHYTHM_NUM_BINS)
  for (let k = 0; k < RHYTHM_NUM_BINS; k++) {
    const fHz = RHYTHM_MIN_HZ + k * RHYTHM_STEP_HZ
    const re  = new Float32Array(N)
    const im  = new Float32Array(N)
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * fHz * n) / fs
      re[n] = hann[n] * Math.cos(angle)
      im[n] = hann[n] * Math.sin(angle)
    }
    kernels[k] = { re, im }
  }
  return kernels
}

/**
 * Compute DFT power at each precomputed kernel frequency.
 * @param {Float32Array} signal
 * @param {Array<{re: Float32Array, im: Float32Array}>} kernels
 * @returns {Float32Array} power per bin
 */
function _dftPower(signal, kernels) {
  const N      = signal.length
  const result = new Float32Array(kernels.length)
  for (let k = 0; k < kernels.length; k++) {
    const { re: reK, im: imK } = kernels[k]
    let r = 0, m = 0
    for (let n = 0; n < N; n++) {
      r += reK[n] * signal[n]
      m += imK[n] * signal[n]
    }
    result[k] = r * r + m * m
  }
  return result
}

/**
 * Z-score normalize a Float32Array. Returns null if std ≈ 0.
 * @param {Float32Array} arr
 * @returns {Float32Array|null}
 */
function _zscore(arr) {
  const N = arr.length
  let sum = 0
  for (let i = 0; i < N; i++) sum += arr[i]
  const mean = sum / N

  let ss = 0
  for (let i = 0; i < N; i++) ss += (arr[i] - mean) ** 2
  const std = Math.sqrt(ss / N)

  if (std < 1e-12) return null

  const out = new Float32Array(N)
  for (let i = 0; i < N; i++) out[i] = (arr[i] - mean) / std
  return out
}
