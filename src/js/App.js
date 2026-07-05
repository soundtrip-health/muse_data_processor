import BPMManager from './managers/BPMManager'
import AudioManager from './managers/AudioManager'
import EEGManager from './managers/EEGManager'
import EntrainmentManager from './managers/EntrainmentManager'
import ComplexityManager from './managers/ComplexityManager'
import RecordingManager from './managers/RecordingManager'
import BioDataDisplay from './ui/BioDataDisplay'

/**
 * App — top-level orchestrator.
 *
 * A Muse EEG/PPG/IMU biometric visualizer. The live bio-data panel is the
 * visualization; there is no 3D scene. Audio playback is optional — loading a
 * local file enables the neural-entrainment analysis (music tempo vs. EEG
 * tempogram). The render loop runs immediately so EEG plots work with no audio.
 */
export default class App {
  //Managers
  static audioManager = null
  static bpmManager = null
  static eegManager = null
  static entrainmentManager = null
  static complexityManager = null
  static recordingManager = null

  constructor() {
    const input = document.getElementById('audio-upload')

    // File upload: loads (or replaces) the buffer audio source.
    input.addEventListener('change', (e) => {
      const file = e.target.files[0]
      if (!file) return
      e.target.value = '' // allow re-selecting the same file
      this._loadAudioFile(file)
    })

    // Mic button: toggle live microphone input as the (muted) audio source.
    this._micBtn = document.getElementById('mic-btn')
    this._micBtn.addEventListener('click', () => this._toggleMic())

    this._setupEEG()
    this._setupRecording()

    // Start the update loop immediately so EEG/bio plots work before (or without) audio.
    this.update()
  }

  /** Lazily create the audio + entrainment managers (idempotent). */
  _ensureAudioInfra() {
    if (!App.audioManager) App.audioManager = new AudioManager()
    if (!App.entrainmentManager) App.entrainmentManager = new EntrainmentManager()
  }

  /**
   * Load a local audio file as the (looping) buffer source, detect its BPM, and
   * play it. Replaces any current source, including live-mic input.
   * @param {File} file — audio File from the upload input
   */
  async _loadAudioFile(file) {
    this._ensureAudioInfra()
    try {
      await App.audioManager.loadAudioBuffer(file)
    } catch (err) {
      console.error('Audio load failed:', err)
      return
    }

    if (!App.bpmManager) App.bpmManager = new BPMManager()
    await App.bpmManager.detectBPM(App.audioManager.audio.buffer)

    App.audioManager.play()
    this._setupPauseBtn()
    this._updateMicButton()
  }

  /** Toggle live microphone input as the (muted, analysis-only) audio source. */
  async _toggleMic() {
    this._ensureAudioInfra()
    const am = App.audioManager

    if (am.isMic) {
      am.stopMic()
    } else {
      this._micBtn.disabled = true
      try {
        await am.startMic()
      } catch (err) {
        // Permission denied or no device — leave the current source untouched.
        console.error('Microphone access failed:', err)
        this._micBtn.disabled = false
        return
      }
      this._micBtn.disabled = false
      // Live input can't be paused — hide the buffer pause button while mic is on.
      if (this._pauseBtn) this._pauseBtn.hidden = true
    }
    this._updateMicButton()
  }

  /** Reflect the current mic state in the mic button + pause button visibility. */
  _updateMicButton() {
    if (!this._micBtn) return
    const active = !!App.audioManager?.isMic
    this._micBtn.classList.toggle('active', active)
    this._micBtn.title = active ? 'Stop microphone input' : 'Use microphone as audio source'
    // Restore the pause button when a buffer source is (again) active.
    if (!active && this._pauseBtn && App.audioManager?.source === 'buffer') {
      this._pauseBtn.hidden = false
    }
  }

  /** Wire up EEG connect/disconnect UI and create the EEGManager instance. */
  _setupEEG() {
    App.eegManager = new EEGManager()
    App.complexityManager = new ComplexityManager()

    const controls    = document.getElementById('eeg-controls')
    const btn         = document.getElementById('eeg-connect')
    const batteryEl   = document.getElementById('eeg-status')
    const batteryFill = batteryEl.querySelector('.battery-fill')
    const panel       = document.getElementById('bio-panel')
    this._hrDisplay   = document.getElementById('heart-rate')
    this._bioHr       = document.getElementById('bio-hr')
    this._qualityDots = document.querySelectorAll('.quality-dot')
    this._bioPanel    = panel
    this._bioVisible  = false

    const updateBattery = (level) => {
      // level: 0–100 or null (disconnected)
      // Vertical battery: fill grows upward inside body (body: y=2.5, height=19, inner max=17)
      const maxH    = 17
      const fillH   = level != null ? Math.round(maxH * level / 100) : 0
      batteryFill.setAttribute('height', fillH)
      batteryFill.setAttribute('y', (21 - fillH).toFixed(1))
      batteryEl.dataset.level = level != null ? level : ''
      batteryEl.title = level != null ? `Battery: ${level}%` : 'Battery'
      // Color: green ≥50, yellow 20–49, red <20, gray when disconnected
      batteryEl.dataset.state = level == null ? 'off'
        : level >= 50 ? 'good'
        : level >= 20 ? 'warn'
        : 'low'
    }
    updateBattery(null)

    controls.style.display = 'flex'

    App.eegManager.onBatteryLevel = (level) => updateBattery(level)

    App.eegManager.onDisconnected = () => {
      btn.textContent = 'Connect EEG'
      btn.disabled = false
      updateBattery(null)
      // Hide the data view and exit full-screen mode.
      panel.hidden = true
      this._bioVisible = false
      document.body.classList.remove('fullscreen-bio')
      if (this._recordBtn) this._recordBtn.hidden = true
    }

    btn.addEventListener('click', async () => {
      if (App.eegManager.isConnected) {
        App.eegManager.disconnect()
      } else {
        btn.textContent = 'Connecting...'
        btn.disabled = true
        try {
          await App.eegManager.connect()
          btn.textContent = 'Disconnect EEG'
          btn.disabled = false
          if (this._recordBtn) this._recordBtn.hidden = false
          // Init plots on first connect; reset read pointers on subsequent connects
          if (!this._bioDisplay) {
            this._bioDisplay = new BioDataDisplay()
            this._bioDisplay.init()
          } else {
            this._bioDisplay.resetIndices()
          }
          // The full-screen data view is the default: show it automatically.
          panel.hidden = false
          this._bioVisible = true
          document.body.classList.add('fullscreen-bio')
        } catch (err) {
          console.error('EEG connect failed:', err)
          btn.textContent = 'Connect EEG'
          btn.disabled = false
        }
      }
    })
  }

  /** Wire up the record button + download-on-stop flow. */
  _setupRecording() {
    App.recordingManager = new RecordingManager()

    const btn     = document.getElementById('bio-record')
    const timeEl  = document.getElementById('bio-record-time')
    this._recordBtn    = btn
    this._recordTimeEl = timeEl

    btn.addEventListener('click', async () => {
      const rm = App.recordingManager
      if (rm.isRecording) {
        btn.disabled = true
        const blob = await rm.stop()
        // Stream mode writes straight to disk and returns null; memory-mode
        // fallback returns a Blob the browser must download.
        if (blob) this._downloadRecording(blob, rm.startedAtMs)
        btn.disabled = false
        btn.classList.remove('active')
        btn.title = 'Record raw data to JSONL'
        timeEl.hidden = true
      } else {
        const eeg = App.eegManager
        const started = await rm.start({
          device:     eeg?.deviceName,
          deviceInfo: eeg?.deviceInfo,
          channels:   ['TP9', 'AF7', 'AF8', 'TP10'],
          audioBpm:   App.bpmManager?.bpmValue || null,
        })
        if (!started) return   // user cancelled the save-file picker
        btn.classList.add('active')
        btn.title = 'Stop recording'
        timeEl.hidden = false
      }
    })
  }

  /** Trigger a browser download of the JSONL blob. */
  _downloadRecording(blob, startedAtMs) {
    const ts    = new Date(startedAtMs).toISOString().replace(/[:.]/g, '-').replace('Z', '')
    const url   = URL.createObjectURL(blob)
    const a     = document.createElement('a')
    a.href      = url
    a.download  = `nouscope-${ts}.jsonl`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  /** Show the pause button and wire its click handler (idempotent). */
  _setupPauseBtn() {
    if (this._pauseBtn) {
      this._pauseBtn.textContent = '⏸'
      return
    }
    this._pauseBtn = document.getElementById('pause-btn')
    this._pauseBtn.addEventListener('click', () => {
      if (App.audioManager.isPlaying) {
        App.audioManager.pause()
        this._pauseBtn.textContent = '▶'
      } else {
        App.audioManager.play()
        this._pauseBtn.textContent = '⏸'
      }
    })
    this._pauseBtn.hidden = false
  }

  /** Update signal quality dots, bio-panel HR readout, and plots each frame. */
  _updateBioPanel() {
    if (!App.eegManager?.isConnected) return

    // Signal quality dots (always update when connected, even if panel hidden)
    const sq = App.eegManager.signalQuality
    this._qualityDots.forEach((dot, i) => {
      dot.classList.remove('good', 'marginal', 'poor')
      dot.classList.add(sq[i])
    })

    // HR in panel
    const hr = App.eegManager.heartRate
    if (this._bioHr) this._bioHr.textContent = hr > 0 ? `${Math.round(hr)} bpm` : ''

    // Plots — only render when panel is visible
    if (this._bioVisible) this._bioDisplay?.update()
  }

  /** Main update loop — called every animation frame. */
  update() {
    requestAnimationFrame(() => this.update())

    App.eegManager?.update(performance.now())

    if (this._hrDisplay && App.eegManager?.isConnected) {
      const hr = App.eegManager.heartRate
      this._hrDisplay.textContent = hr > 0 ? `♥ ${Math.round(hr)} bpm` : ''
    }

    this._updateBioPanel()

    App.audioManager?.update()
    App.entrainmentManager?.update(performance.now())
    App.complexityManager?.update(performance.now())

    // Record-time readout (updates ~once per second is plenty; cheap per-frame)
    if (this._recordTimeEl && App.recordingManager?.isRecording) {
      const s = Math.floor(App.recordingManager.elapsedMs() / 1000)
      const mm = Math.floor(s / 60).toString().padStart(2, '0')
      const ss = (s % 60).toString().padStart(2, '0')
      this._recordTimeEl.textContent = `${mm}:${ss}`
    }
  }
}
