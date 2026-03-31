import * as THREE from 'three'

// --- Constants ---
const FFT_SIZE = 1024
const HALF_FFT = FFT_SIZE / 2          // 512 magnitude bins
const NOVELTY_RING_LEN = 768           // ~12.8 s at 60 fps — exceeds 8 s analysis window

const FREQ_BANDS = {
  lowMin: 10,    // Hz — low band start
  lowMax: 250,   // Hz — low band end / mid band start
  midMax: 2000,  // Hz — mid band end / high band start
  highMax: 20000 // Hz — high band end
}

/**
 * AudioManager
 *
 * Wraps Three.js AudioListener / AudioAnalyser to load, play, and analyze
 * audio. Produces normalized frequency band data (low / mid / high, 0–1)
 * that ReactiveParticles maps to shader uniforms each frame.
 *
 * Usage:
 *   const mgr = new AudioManager()
 *   await mgr.loadAudioBuffer(file)   // File object from <input>
 *   await mgr.loadAudioBuffer('/audio/demo.mp3')  // URL string
 *   mgr.play()
 *   // each frame:
 *   mgr.update()
 *   // read: mgr.frequencyData.low / .mid / .high  (0–1)
 */
export default class AudioManager {
  constructor() {
    this.frequencyArray = []
    this.frequencyData = {
      low: 0,
      mid: 0,
      high: 0,
    }
    this.isPlaying = false
    this.smoothedLowFrequency = 0
    this.audioContext = null

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

  /**
   * Load and decode an audio source.
   * @param {File|string} source — a File object (from <input type="file">)
   *   or a URL string (path to audio file)
   * @returns {Promise<void>} resolves when buffer is ready for playback
   */
  async loadAudioBuffer(source) {
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

  /** Start playback. */
  play() {
    this.audio.play()
    this.isPlaying = true
  }

  /** Pause playback. */
  pause() {
    this.audio.pause()
    this.isPlaying = false
  }

  collectAudioData() {
    this.frequencyArray = this.audioAnalyser.getFrequencyData()
  }

  analyzeFrequency() {
    // Compute bin indices from Hz values using the FFT size and sample rate
    const lowStart  = Math.floor((FREQ_BANDS.lowMin  * this.bufferLength) / this.audioContext.sampleRate)
    const lowEnd    = Math.floor((FREQ_BANDS.lowMax   * this.bufferLength) / this.audioContext.sampleRate)
    const midStart  = Math.floor((FREQ_BANDS.lowMax   * this.bufferLength) / this.audioContext.sampleRate)
    const midEnd    = Math.floor((FREQ_BANDS.midMax   * this.bufferLength) / this.audioContext.sampleRate)
    const highStart = Math.floor((FREQ_BANDS.midMax   * this.bufferLength) / this.audioContext.sampleRate)
    const highEnd   = this.bufferLength - 1

    const lowAvg  = this.normalizeValue(this.calculateAverage(this.frequencyArray, lowStart,  lowEnd))
    const midAvg  = this.normalizeValue(this.calculateAverage(this.frequencyArray, midStart,  midEnd))
    const highAvg = this.normalizeValue(this.calculateAverage(this.frequencyArray, highStart, highEnd))

    this.frequencyData = {
      low:  lowAvg,
      mid:  midAvg,
      high: highAvg,
    }
  }

  calculateAverage(array, start, end) {
    let sum = 0
    for (let i = start; i <= end; i++) {
      sum += array[i]
    }
    return sum / (end - start + 1)
  }

  normalizeValue(value) {
    // 0–255 (8-bit unsigned byte data from AnalyserNode)
    return value / 255
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

    this.collectAudioData()
    this.analyzeFrequency()
    this._sampleNovelty()
  }
}
