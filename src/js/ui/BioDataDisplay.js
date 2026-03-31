import { WebglLineRoll, ColorRGBA, createWebGL2Context, setBackgroundColor } from 'webgl-plot'
import App from '../App'

// ── Constants ─────────────────────────────────────────────────────────────────

const EEG_ROLL      = 512   // 2 s at 256 Hz
const EEG_SCALE     = 200   // µV mapped to full per-channel half-height
const EEG_AMPLITUDE = 0.22  // normalized half-height per channel stripe
const EEG_OFFSETS   = [0.75, 0.25, -0.25, -0.75]  // TP9, AF7, AF8, TP10

const BAND_ROLL        = 300   // ~5 s at ~60 fps polling rate
const BAND_YMAX_MIN    = 0.2   // minimum visible Y range (prevents zero-range when all flat)
const BAND_YMAX_DECAY  = 0.999 // per-frame running-max decay (~6% per 100 frames)
const BAND_LERP        = 0.08  // per-frame EMA toward bandPower (~0.5 s to 90% at 60 fps)

const PPG_ROLL = 384  // 6 s at 64 Hz — matches MSPTD analysis window

const IMU_ROLL    = 208    // 4 s at ~52 Hz
const ACCEL_SCALE = 2.0    // ±2 g (Muse full range)
const GYRO_SCALE  = 300    // ±300 dps

// ── Colors (0–255) ────────────────────────────────────────────────────────────

const EEG_COLORS = [
  new ColorRGBA(77,  217, 255, 255),   // TP9  — cyan
  new ColorRGBA(102, 255, 128, 255),   // AF7  — green
  new ColorRGBA(255, 166,  51, 255),   // AF8  — orange
  new ColorRGBA(229, 102, 255, 255),   // TP10 — magenta
]

// delta=violet, theta=blue, alpha=green, beta=amber, gamma=red
const BAND_COLORS = [
  new ColorRGBA(167, 139, 250, 255),  // δ delta  — violet
  new ColorRGBA( 96, 165, 250, 255),  // θ theta  — blue
  new ColorRGBA( 52, 211, 153, 255),  // α alpha  — green
  new ColorRGBA(251, 191,  36, 255),  // β beta   — amber
  new ColorRGBA(248, 113, 113, 255),  // γ gamma  — red
]

const PPG_COLOR = new ColorRGBA(255, 128, 128, 255)  // salmon

const IMU_COLORS = [
  new ColorRGBA( 77, 178, 255, 255),   // ax — blue
  new ColorRGBA( 51, 128, 230, 255),   // ay — darker blue
  new ColorRGBA(128, 210, 255, 255),   // az — lighter blue
  new ColorRGBA(255, 102, 102, 255),   // gx — red
  new ColorRGBA(217,  77,  77, 255),   // gy — darker red
  new ColorRGBA(255, 153, 128, 255),   // gz — lighter red
]

// ── Spectrogram ──────────────────────────────────────────────────────────────

const SPEC_BINS        = 50
const SPEC_START_BIN   = 8     // first Hz to display in main spec (1–7 Hz shown in lo-spec only)
const SPEC_START_IDX   = SPEC_START_BIN - 1              // array index for SPEC_START_BIN
const SPEC_DISPLAY_BINS = SPEC_BINS - SPEC_START_IDX     // bins shown: 8–50 Hz = 43
const SPEC_PX_PER_BIN  = 2     // vertical pixels per frequency bin
const SPEC_COL_WIDTH   = 2     // horizontal pixels per time column (scroll speed)
const SPEC_SCALE_DECAY = 0.995   // running floor decay per column (keeps noise floor from anchoring)
const SPEC_MIN_RANGE   = 2      // minimum log₁₀ dynamic range
const SPEC_SCALE_WIN   = 30     // columns in sliding window for robust ceiling (~15 s at 2 Hz)
const SPEC_SCALE_PCT   = 0.9    // percentile of window used as scale ceiling
// log₁₀ cap ≈ 200 µV amplitude (Hann-DFT power ≈ A² × N²/16; log₁₀(200² × 4096) ≈ 8.2)
const SPEC_LOG_CAP     = 8.2

// Low-end spectrogram (0.5–8.0 Hz at 0.1 Hz resolution, hi-res entrainment view)
const SPEC_LO_BINS       = 76  // 0.5–8.0 Hz at 0.1 Hz steps
const SPEC_LO_PX_PER_BIN = 1   // 1 pixel per bin → 76 px tall

// Viridis colormap — piecewise-linear from 9 key stops, precomputed to 256-entry LUT
const _VIRIDIS_STOPS = [
  [0.000,  68,   1,  84],
  [0.125,  72,  36, 117],
  [0.250,  59,  82, 139],
  [0.375,  44, 114, 142],
  [0.500,  33, 145, 140],
  [0.625,  57, 175, 120],
  [0.750, 122, 209,  81],
  [0.875, 189, 223,  38],
  [1.000, 253, 231,  37],
]

const VIRIDIS_LUT = (() => {
  const lut = new Array(256)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let seg = 1
    while (seg < _VIRIDIS_STOPS.length - 1 && t > _VIRIDIS_STOPS[seg][0]) seg++
    const [t0, r0, g0, b0] = _VIRIDIS_STOPS[seg - 1]
    const [t1, r1, g1, b1] = _VIRIDIS_STOPS[seg]
    const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0
    lut[i] = [
      Math.round(r0 + f * (r1 - r0)),
      Math.round(g0 + f * (g1 - g0)),
      Math.round(b0 + f * (b1 - b0)),
    ]
  }
  return lut
})()

// ─────────────────────────────────────────────────────────────────────────────

const TRANSPARENT = [0, 0, 0, 0]

// ─────────────────────────────────────────────────────────────────────────────

export default class BioDataDisplay {
  _eegPlot  = null
  _bandPlot = null
  _ppgPlot  = null
  _imuPlot  = null

  _eegGL   = null
  _bandGL  = null
  _ppgGL   = null
  _imuGL   = null
  _specCtx      = null  // 2D canvas context for full spectrogram (1–50 Hz)
  _specLoCtx    = null  // 2D canvas context for low-end spectrogram (1–8 Hz)
  _specAudioCtx = null  // 2D canvas context for audio tempogram (0.5–5 Hz)

  _eegReadIdx = [0, 0, 0, 0]
  _ppgReadIdx = 0
  _imuReadIdx = 0

  _ppgPeak   = 1              // running peak abs value for PPG auto-scale
  _bandYMax  = BAND_YMAX_MIN  // running peak for band auto-scale
  _bandSmoothed = [0, 0, 0, 0, 0]  // per-frame EMA of band values (avoids staircase)
  _bandItemEls = null         // [δ, θ, α, β, γ] legend row <div> references
  _bandValEls  = null         // [δ, θ, α, β, γ] value <span> references

  _specReadIdx   = 0
  _specLoReadIdx = 0
  _specLo      = 0            // running floor of log₁₀ power scale (full); decays upward slowly
  _specHiWin   = []           // sliding window of per-column max for percentile ceiling (full)
  _specLoLo    = 0            // running floor for low-freq view
  _specLoHiWin = []           // sliding window of per-column max for percentile ceiling (lo-freq)

  // Audio tempogram display state
  _specAudioLo    = 0
  _specAudioHiWin = []
  _entrainValueEl = null
  _entrainFillEl  = null

  /**
   * Create WebGL contexts and rolling-line plots for all three canvases.
   * Safe to call while the panel is hidden — canvas dimensions are set via HTML attributes.
   */
  init() {
    // EEG — 4 stacked channels
    const eegCanvas = document.getElementById('eeg-canvas')
    this._eegGL = createWebGL2Context(eegCanvas, { transparent: true })
    setBackgroundColor(this._eegGL, TRANSPARENT)
    this._eegPlot = new WebglLineRoll(this._eegGL, EEG_ROLL, 4)
    EEG_COLORS.forEach((c, i) => this._eegPlot.setLineColor(c, i))

    // Spectrogram — 2D canvas (not WebGL), scrolling heatmap of Hann-DFT power
    const specCanvas = document.getElementById('spec-canvas')
    this._specCtx = specCanvas.getContext('2d')
    this._specCtx.fillStyle = '#000'
    this._specCtx.fillRect(0, 0, specCanvas.width, specCanvas.height)

    // Low-end spectrogram (1–8 Hz, beat entrainment view)
    const specLoCanvas = document.getElementById('spec-lo-canvas')
    this._specLoCtx = specLoCanvas.getContext('2d')
    this._specLoCtx.fillStyle = '#000'
    this._specLoCtx.fillRect(0, 0, specLoCanvas.width, specLoCanvas.height)

    // Audio tempogram (0.5–5.0 Hz spectral-flux novelty, entrainment comparison)
    const specAudioCanvas = document.getElementById('spec-audio-canvas')
    this._specAudioCtx = specAudioCanvas.getContext('2d')
    this._specAudioCtx.fillStyle = '#000'
    this._specAudioCtx.fillRect(0, 0, specAudioCanvas.width, specAudioCanvas.height)
    this._entrainValueEl = document.getElementById('bio-entrain-value')
    this._entrainFillEl  = document.getElementById('bio-entrain-fill')

    // EEG band powers — 5 lines (delta, theta, alpha, beta, gamma), polled each frame
    const bandCanvas = document.getElementById('band-canvas')
    this._bandGL = createWebGL2Context(bandCanvas, { transparent: true })
    setBackgroundColor(this._bandGL, TRANSPARENT)
    this._bandPlot = new WebglLineRoll(this._bandGL, BAND_ROLL, 5)
    BAND_COLORS.forEach((c, i) => this._bandPlot.setLineColor(c, i))

    const BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    this._bandItemEls = BAND_NAMES.map(b => document.getElementById(`band-item-${b}`))
    this._bandValEls  = BAND_NAMES.map(b => document.getElementById(`band-val-${b}`))

    // PPG — single filtered infrared trace
    const ppgCanvas = document.getElementById('ppg-canvas')
    this._ppgGL = createWebGL2Context(ppgCanvas, { transparent: true })
    setBackgroundColor(this._ppgGL, TRANSPARENT)
    this._ppgPlot = new WebglLineRoll(this._ppgGL, PPG_ROLL, 1)
    this._ppgPlot.setLineColor(PPG_COLOR, 0)

    // IMU — accel (3) + gyro (3) in one plot
    const imuCanvas = document.getElementById('imu-canvas')
    this._imuGL = createWebGL2Context(imuCanvas, { transparent: true })
    setBackgroundColor(this._imuGL, TRANSPARENT)
    this._imuPlot = new WebglLineRoll(this._imuGL, IMU_ROLL, 6)
    IMU_COLORS.forEach((c, i) => this._imuPlot.setLineColor(c, i))

    this.resetIndices()
  }

  /** Reset read pointers to the current sample counts — call on each reconnect. */
  resetIndices() {
    const mgr = App.eegManager
    if (!mgr) return
    this._eegReadIdx = [mgr.eegSampleCount, mgr.eegSampleCount, mgr.eegSampleCount, mgr.eegSampleCount]
    this._ppgReadIdx = mgr.ppgSampleCount
    this._imuReadIdx = mgr.imuSampleCount
    this._ppgPeak    = 1
    this._bandYMax   = BAND_YMAX_MIN
    this._bandSmoothed = [0, 0, 0, 0, 0]
    this._specReadIdx   = mgr.spectrumSampleCount
    this._specLoReadIdx = mgr.spectrumLoSampleCount
    this._specLo      = 0
    this._specHiWin   = []
    this._specLoLo    = 0
    this._specLoHiWin = []
  }

  /** Call each animation frame (only when panel is visible). */
  update() {
    const mgr = App.eegManager
    if (mgr?.isConnected) {
      this._updateEEG(mgr)
      this._updateSpectrum(mgr)
      this._updateSpectrumLo(mgr)
      this._updateBands(mgr)
      this._updatePPG(mgr)
      this._updateIMU(mgr)
    }
    // Audio tempogram + entrainment meter update even without EEG
    this._updateAudioTempogram()
    this._updateEntrainmentMeter()
  }

  // ── Private ──────────────────────────────────────────────────────────────────

  _updateEEG(mgr) {
    const chs = mgr.eegChannels
    // available = new samples produced since last read, capped at buffer capacity
    const available = Math.min(mgr.eegSampleCount - this._eegReadIdx[0], chs[0].length)
    if (available <= 0) return

    // The new samples occupy the last `available` slots of each rolling buffer
    const start = chs[0].length - available
    for (let s = 0; s < available; s++) {
      const yVals = EEG_OFFSETS.map((offset, ch) => {
        const v = chs[ch][start + s] ?? 0
        return offset + (v / EEG_SCALE) * EEG_AMPLITUDE
      })
      this._eegPlot.addPoint(yVals)
    }
    this._eegReadIdx = [mgr.eegSampleCount, mgr.eegSampleCount, mgr.eegSampleCount, mgr.eegSampleCount]

    this._eegGL.clear(this._eegGL.COLOR_BUFFER_BIT)
    this._eegPlot.draw()
  }

  _updateSpectrum(mgr) {
    const buf = mgr.spectrumDisplay
    const available = Math.min(mgr.spectrumSampleCount - this._specReadIdx, buf.length)
    if (available <= 0) return

    const canvas = this._specCtx.canvas
    const H = SPEC_DISPLAY_BINS * SPEC_PX_PER_BIN   // 43 × 2 = 86
    const start = buf.length - available

    for (let s = 0; s < available; s++) {
      const col = buf[start + s]

      // Shift existing content left by SPEC_COL_WIDTH pixels
      this._specCtx.drawImage(canvas, -SPEC_COL_WIDTH, 0)

      const imgData = this._specCtx.createImageData(SPEC_COL_WIDTH, H)

      // Auto-scale: 90th-percentile window for ceiling (artifact-robust), running min+decay for floor
      let colMax = -Infinity
      for (let i = SPEC_START_IDX; i < SPEC_BINS; i++) {
        if (col[i] > -9) {   // skip near-zero padding
          if (col[i] > colMax) colMax = col[i]
          if (col[i] < this._specLo) this._specLo = col[i]
        }
      }
      if (colMax > -Infinity) {
        this._specHiWin.push(Math.min(colMax, SPEC_LOG_CAP))
        if (this._specHiWin.length > SPEC_SCALE_WIN) this._specHiWin.shift()
      }
      this._specLo += (1 - SPEC_SCALE_DECAY)

      const sorted = this._specHiWin.slice().sort((a, b) => a - b)
      const specHi = sorted.length
        ? sorted[Math.max(0, Math.floor((sorted.length - 1) * SPEC_SCALE_PCT))]
        : this._specLo + SPEC_MIN_RANGE
      const range = Math.max(specHi - this._specLo, SPEC_MIN_RANGE)

      // Draw new column at right edge — each bin spans SPEC_PX_PER_BIN rows
      for (let i = SPEC_START_IDX; i < SPEC_BINS; i++) {
        const norm = (col[i] - this._specLo) / range
        const idx = Math.max(0, Math.min(255, Math.round(norm * 255)))
        const [r, g, b] = VIRIDIS_LUT[idx]
        const binInDisplay = i - SPEC_START_IDX
        for (let p = 0; p < SPEC_PX_PER_BIN; p++) {
          const y = H - 1 - (binInDisplay * SPEC_PX_PER_BIN + p)
          for (let cx = 0; cx < SPEC_COL_WIDTH; cx++) {
            const off = (y * SPEC_COL_WIDTH + cx) * 4
            imgData.data[off]     = r
            imgData.data[off + 1] = g
            imgData.data[off + 2] = b
            imgData.data[off + 3] = 255
          }
        }
      }

      this._specCtx.putImageData(imgData, canvas.width - SPEC_COL_WIDTH, 0)
    }

    this._specReadIdx = mgr.spectrumSampleCount
  }

  _updateSpectrumLo(mgr) {
    const buf = mgr.spectrumLoDisplay
    const available = Math.min(mgr.spectrumLoSampleCount - this._specLoReadIdx, buf.length)
    if (available <= 0) return

    const canvas = this._specLoCtx.canvas
    const H = SPEC_LO_BINS * SPEC_LO_PX_PER_BIN   // 76 × 1 = 76
    const start = buf.length - available

    for (let s = 0; s < available; s++) {
      const col = buf[start + s]

      // Shift existing content left by SPEC_COL_WIDTH pixels
      this._specLoCtx.drawImage(canvas, -SPEC_COL_WIDTH, 0)

      const imgData = this._specLoCtx.createImageData(SPEC_COL_WIDTH, H)

      // Auto-scale: 90th-percentile window for ceiling (artifact-robust), running min+decay for floor
      let colMax = -Infinity
      for (let i = 0; i < SPEC_LO_BINS; i++) {
        if (col[i] > -9) {
          if (col[i] > colMax) colMax = col[i]
          if (col[i] < this._specLoLo) this._specLoLo = col[i]
        }
      }
      if (colMax > -Infinity) {
        this._specLoHiWin.push(Math.min(colMax, SPEC_LOG_CAP))
        if (this._specLoHiWin.length > SPEC_SCALE_WIN) this._specLoHiWin.shift()
      }
      this._specLoLo += (1 - SPEC_SCALE_DECAY)

      const sortedLo = this._specLoHiWin.slice().sort((a, b) => a - b)
      const specLoHi = sortedLo.length
        ? sortedLo[Math.max(0, Math.floor((sortedLo.length - 1) * SPEC_SCALE_PCT))]
        : this._specLoLo + SPEC_MIN_RANGE
      const range = Math.max(specLoHi - this._specLoLo, SPEC_MIN_RANGE)

      // Draw new column at right edge — 1 pixel per bin
      for (let i = 0; i < SPEC_LO_BINS; i++) {
        const norm = (col[i] - this._specLoLo) / range
        const idx = Math.max(0, Math.min(255, Math.round(norm * 255)))
        const [r, g, b] = VIRIDIS_LUT[idx]
        const y = H - 1 - i
        for (let cx = 0; cx < SPEC_COL_WIDTH; cx++) {
          const off = (y * SPEC_COL_WIDTH + cx) * 4
          imgData.data[off]     = r
          imgData.data[off + 1] = g
          imgData.data[off + 2] = b
          imgData.data[off + 3] = 255
        }
      }

      this._specLoCtx.putImageData(imgData, canvas.width - SPEC_COL_WIDTH, 0)
    }

    this._specLoReadIdx = mgr.spectrumLoSampleCount
  }

  _updateBands(mgr) {
    const { delta, theta, alpha, beta, gamma } = mgr.bandPower
    const target = [delta, theta, alpha, beta, gamma]

    // Per-frame EMA toward bandPower — smooths the ~2 Hz staircase updates into
    // continuous curves suitable for 60 fps display.
    for (let i = 0; i < 5; i++) {
      this._bandSmoothed[i] += BAND_LERP * (target[i] - this._bandSmoothed[i])
    }

    // Auto-scale: running max with slow decay, floored at BAND_YMAX_MIN so the
    // plot always shows at least some vertical range even when all bands are flat.
    const frameMax = Math.max(...this._bandSmoothed)
    this._bandYMax = Math.max(this._bandYMax * BAND_YMAX_DECAY, frameMax, BAND_YMAX_MIN)

    // Map [0, _bandYMax] → [-1, +1] for webgl-plot (0 = bottom, peak = top)
    this._bandPlot.addPoint(this._bandSmoothed.map(v => v / this._bandYMax * 2 - 1))
    this._bandGL.clear(this._bandGL.COLOR_BUFFER_BIT)
    this._bandPlot.draw()

    // Show legend rows only for bands mapped to a viz parameter
    const normBands  = mgr.normalizeBands
    const BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    for (let i = 0; i < 5; i++) {
      const active = normBands.has(BAND_NAMES[i])
      this._bandItemEls[i].style.display = active ? '' : 'none'
      if (active) this._bandValEls[i].textContent = this._bandSmoothed[i].toFixed(2)
    }
  }

  _updatePPG(mgr) {
    const buf = mgr.ppgDisplay
    const available = Math.min(mgr.ppgSampleCount - this._ppgReadIdx, buf.length)
    if (available <= 0) return

    // Update auto-scale from recent peak magnitude
    const recent = buf.slice(-128)
    const peak = Math.max(...recent.map(Math.abs))
    if (peak > 0) this._ppgPeak = peak * 1.2

    const start = buf.length - available
    for (let s = 0; s < available; s++) {
      this._ppgPlot.addPoint([buf[start + s] / this._ppgPeak])
    }
    this._ppgReadIdx = mgr.ppgSampleCount

    this._ppgGL.clear(this._ppgGL.COLOR_BUFFER_BIT)
    this._ppgPlot.draw()
  }

  _updateIMU(mgr) {
    const acc = mgr.accelDisplay
    const gyr = mgr.gyroDisplay
    const available = Math.min(mgr.imuSampleCount - this._imuReadIdx, acc.x.length)
    if (available <= 0) return

    const start = acc.x.length - available
    // Gyro buffer may lag by a sample or two — clamp index to its length
    const gStart = Math.max(0, gyr.x.length - available)
    for (let s = 0; s < available; s++) {
      const gi = Math.min(gStart + s, gyr.x.length - 1)
      this._imuPlot.addPoint([
        (acc.x[start + s] ?? 0) / ACCEL_SCALE,
        (acc.y[start + s] ?? 0) / ACCEL_SCALE,
        (acc.z[start + s] ?? 0) / ACCEL_SCALE,
        (gyr.x[gi] ?? 0) / GYRO_SCALE,
        (gyr.y[gi] ?? 0) / GYRO_SCALE,
        (gyr.z[gi] ?? 0) / GYRO_SCALE,
      ])
    }
    this._imuReadIdx = mgr.imuSampleCount

    this._imuGL.clear(this._imuGL.COLOR_BUFFER_BIT)
    this._imuPlot.draw()
  }

  // ── Audio tempogram heatmap ────────────────────────────────────────────────

  _updateAudioTempogram() {
    const eMgr = App.entrainmentManager
    const spec = eMgr?.audioTempogram
    if (!spec) return

    const BINS      = spec.length    // 46
    const COL_WIDTH = SPEC_COL_WIDTH // 2 px
    const canvas    = this._specAudioCtx.canvas
    const H         = BINS           // 1 px per bin → 46 px tall

    // Shift existing content left
    this._specAudioCtx.drawImage(canvas, -COL_WIDTH, 0)

    const imgData = this._specAudioCtx.createImageData(COL_WIDTH, H)

    // Convert to log scale for display (consistent with other spectrograms)
    let colMax = -Infinity
    for (let i = 0; i < BINS; i++) {
      const logVal = Math.log10(spec[i] + 1e-10)
      if (logVal > colMax) colMax = logVal
      if (logVal < this._specAudioLo) this._specAudioLo = logVal
    }
    if (colMax > -Infinity) {
      this._specAudioHiWin.push(colMax)
      if (this._specAudioHiWin.length > SPEC_SCALE_WIN) this._specAudioHiWin.shift()
    }
    this._specAudioLo += (1 - SPEC_SCALE_DECAY)

    const sorted = this._specAudioHiWin.slice().sort((a, b) => a - b)
    const hi = sorted.length
      ? sorted[Math.max(0, Math.floor((sorted.length - 1) * SPEC_SCALE_PCT))]
      : this._specAudioLo + SPEC_MIN_RANGE
    const range = Math.max(hi - this._specAudioLo, SPEC_MIN_RANGE)

    for (let i = 0; i < BINS; i++) {
      const logVal = Math.log10(spec[i] + 1e-10)
      const norm = (logVal - this._specAudioLo) / range
      const idx  = Math.max(0, Math.min(255, Math.round(norm * 255)))
      const [r, g, b] = VIRIDIS_LUT[idx]
      const y = H - 1 - i
      for (let cx = 0; cx < COL_WIDTH; cx++) {
        const off = (y * COL_WIDTH + cx) * 4
        imgData.data[off]     = r
        imgData.data[off + 1] = g
        imgData.data[off + 2] = b
        imgData.data[off + 3] = 255
      }
    }

    this._specAudioCtx.putImageData(imgData, canvas.width - COL_WIDTH, 0)
  }

  // ── Entrainment meter ──────────────────────────────────────────────────────

  _updateEntrainmentMeter() {
    const eMgr = App.entrainmentManager
    if (!eMgr || !this._entrainFillEl) return

    const val = eMgr.entrainment
    this._entrainFillEl.style.width = `${(val * 100).toFixed(1)}%`

    if (this._entrainValueEl) {
      this._entrainValueEl.textContent = val > 0.01 ? `${(val * 100).toFixed(0)}%` : ''
    }
  }
}
