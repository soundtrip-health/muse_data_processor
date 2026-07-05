import * as THREE from 'three'

// --- Constants ---
const FFT_SIZE = 1024
const HALF_FFT = FFT_SIZE / 2          // 512 magnitude bins
const NOVELTY_RING_LEN = 768           // ~12.8 s at 60 fps — exceeds 8 s analysis window

// Raw capture — disable browser DSP so the spectral content reaching the
// novelty analysis is unmodified (echo cancellation / AGC would distort it).
const MIC_CONSTRAINTS = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
}

/**
 * AudioManager
 *
 * Owns the audio novelty pipeline that feeds EntrainmentManager. The novelty
 * curve can be driven by one of two mutually-exclusive sources:
 *   - 'buffer' — a decoded local audio file, played (looped) via THREE.Audio
 *   - 'mic'    — live microphone input, captured muted (analysis only)
 * Either way, `update()` computes a spectral-flux novelty sample per frame into
 * a shared timestamped ring buffer.
 *
 * Usage:
 *   const mgr = new AudioManager()
 *   await mgr.loadAudioBuffer(file); mgr.play()   // file source
 *   await mgr.startMic()                          // live-mic source (muted)
 *   // each frame:
 *   mgr.update()
 *   // read: mgr.noveltyRing
 */
export default class AudioManager {
  constructor() {
    this.frequencyArray = []
    this.isPlaying = false
    this.audioContext = null
    this.source = null            // 'buffer' | 'mic' | null — which input feeds novelty

    // Live-mic capture
    this._micStream   = null      // MediaStream from getUserMedia
    this._micSource   = null      // MediaStreamAudioSourceNode
    this._micAnalyser = null      // AnalyserNode
    this._micFreq     = new Uint8Array(HALF_FFT)

    // Spectral flux novelty curve (for entrainment tempogram)
    this._prevMagnitudes = new Float32Array(HALF_FFT)
    this._noveltyRing = {
      times:  new Float64Array(NOVELTY_RING_LEN),
      values: new Float32Array(NOVELTY_RING_LEN),
      head:   0,
      count:  0,
    }
  }

  /** Ring buffer of (timestamp, spectralFlux) pairs for EntrainmentManager. */
  get noveltyRing() { return this._noveltyRing }

  /** True while live microphone input is the active novelty source. */
  get isMic() { return this.source === 'mic' }

  /**
   * Load and decode an audio source into a looping buffer. Replaces any active
   * live-mic source. Call play() afterwards to start playback + analysis.
   * @param {File|string} source — a File object (from <input type="file">)
   *   or a URL string (path to audio file)
   * @returns {Promise<void>} resolves when buffer is ready for playback
   */
  async loadAudioBuffer(source) {
    // A file replaces any live-mic source; stop everything so update() idles
    // (isPlaying=false) until play() starts the new buffer.
    this._teardownMic()
    if (this.audio && this.isPlaying) { try { this.audio.stop() } catch {} }
    this.isPlaying = false
    this.source = null

    const audioListener = new THREE.AudioListener()
    this.audio = new THREE.Audio(audioListener)
    this.audioAnalyser = new THREE.AudioAnalyser(this.audio, FFT_SIZE)

    let arrayBuffer

    if (source instanceof File) {
      // Read File object into ArrayBuffer
      arrayBuffer = await source.arrayBuffer()
    } else {
      // Fetch URL string
      const response = await fetch(source)
      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`)
      }
      arrayBuffer = await response.arrayBuffer()
    }

    const audioContext = this.audioAnalyser.analyser.context
    const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer)

    this.audio.setBuffer(decodedBuffer)
    this.audio.setLoop(true)
    this.audio.setVolume(0.5)
    this.audioContext = audioContext
    this.bufferLength = this.audioAnalyser.data.length
  }

  /** Start playback of the loaded buffer. */
  play() {
    this.audio.play()
    this.isPlaying = true
    this.source = 'buffer'
  }

  /** Pause buffer playback (no effect on live-mic input). */
  pause() {
    this.audio.pause()
    this.isPlaying = false
  }

  /**
   * Begin live microphone capture as the novelty source. The mic is routed to
   * an AnalyserNode only — never to the audio destination — so nothing is
   * played back (analysis only, no feedback). Replaces any file source.
   * @returns {Promise<void>}
   */
  async startMic() {
    // Prompt for mic permission first; a rejection leaves the current source intact.
    const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS)

    // Stop any file playback so only one source feeds the analysis.
    if (this.audio && this.source === 'buffer') { try { this.audio.stop() } catch {} }

    const ctx = this._ensureContext()
    if (ctx.state === 'suspended') await ctx.resume()

    this._micStream   = stream
    this._micSource   = ctx.createMediaStreamSource(stream)
    this._micAnalyser = ctx.createAnalyser()
    this._micAnalyser.fftSize = FFT_SIZE
    // Connect source → analyser only; NOT to ctx.destination, so it stays muted.
    this._micSource.connect(this._micAnalyser)

    this.bufferLength = this._micAnalyser.frequencyBinCount   // 512
    this.source = 'mic'
    this.isPlaying = true
  }

  /** Stop live-mic capture and release the microphone. */
  stopMic() {
    this._teardownMic()
    if (this.source === 'mic') {
      this.source = null
      this.isPlaying = false
    }
  }

  /** Disconnect and release mic nodes/tracks (idempotent). */
  _teardownMic() {
    if (this._micSource) { try { this._micSource.disconnect() } catch {} }
    if (this._micStream) this._micStream.getTracks().forEach((t) => t.stop())
    this._micStream = this._micSource = this._micAnalyser = null
  }

  /** Lazily create a standalone AudioContext (for mic use before any file loads). */
  _ensureContext() {
    if (!this.audioContext) {
      const Ctx = window.AudioContext || window.webkitAudioContext
      this.audioContext = new Ctx()
    }
    return this.audioContext
  }

  collectAudioData() {
    this.frequencyArray = this.audioAnalyser.getFrequencyData()
  }

  /**
   * Compute half-wave rectified spectral flux from successive FFT frames.
   * One novelty sample per render frame (~60 Hz), stored in a timestamped
   * ring buffer for EntrainmentManager to resample to a uniform grid.
   */
  _sampleNovelty() {
    const freq = this.frequencyArray
    let flux = 0
    for (let i = 0; i < HALF_FFT; i++) {
      const diff = freq[i] - this._prevMagnitudes[i]
      if (diff > 0) flux += diff
      this._prevMagnitudes[i] = freq[i]
    }

    const ring = this._noveltyRing
    const idx = ring.head
    ring.times[idx]  = performance.now() / 1000   // seconds
    ring.values[idx] = flux
    ring.head  = (idx + 1) % NOVELTY_RING_LEN
    ring.count = Math.min(ring.count + 1, NOVELTY_RING_LEN)
  }

  /** Called every frame from App.update(). */
  update() {
    if (!this.isPlaying) return

    if (this.source === 'mic') {
      this._micAnalyser.getByteFrequencyData(this._micFreq)
      this.frequencyArray = this._micFreq
    } else {
      this.collectAudioData()
    }
    this._sampleNovelty()
  }
}
