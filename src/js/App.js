import * as THREE from 'three'
import ReativeParticles from './entities/ReactiveParticles'
import * as dat from 'dat.gui'
import BPMManager from './managers/BPMManager'
import AudioManager from './managers/AudioManager'
import EEGManager from './managers/EEGManager'
import EntrainmentManager from './managers/EntrainmentManager'
import BioDataDisplay from './ui/BioDataDisplay'
import JellyfinManager from './managers/JellyfinManager'
import JellyfinBrowser from './ui/JellyfinBrowser'

const DEMO_TRACK_URL = './audio/demo.mp3'

/**
 * App — top-level orchestrator.
 *
 * Manages the Three.js scene, camera, renderer, and all manager instances.
 * On user interaction (click or file upload), loads audio, detects BPM,
 * then starts the particle visualizer and render loop.
 */
export default class App {
  //THREE objects
  static holder = null
  static gui = null

  //Managers
  static audioManager = null
  static bpmManager = null
  static eegManager = null
  static entrainmentManager = null

  constructor() {
    const overlay = document.querySelector('.user_interaction')
    const input = document.getElementById('audio-upload')

    // File upload: first load initializes everything; subsequent uploads swap the track
    input.addEventListener('change', (e) => {
      const file = e.target.files[0]
      if (!file) return
      e.target.value = '' // allow re-selecting the same file
      if (this.renderer) {
        this._swapAudio(file)
      } else {
        overlay.removeEventListener('click', this._overlayClickHandler)
        this.init(file)
      }
    })

    // Click anywhere on overlay: load demo track
    this._overlayClickHandler = (e) => {
      overlay.removeEventListener('click', this._overlayClickHandler)
      this.init(DEMO_TRACK_URL)
    }
    overlay.addEventListener('click', this._overlayClickHandler)

    this._setupEEG()
    this._setupJellyfin()

    // Start the render/update loop immediately so EEG plots work before music starts
    this.update()
  }

  /**
   * Initialize scene, renderer, camera and kick off audio loading.
   * @param {File|string} source — File object or URL string for the audio
   */
  init(source) {
    document.querySelector('.user_interaction__label').textContent = 'Loading...'

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    })

    this.renderer.setClearColor(0x000000, 0)
    this.renderer.setSize(window.innerWidth, window.innerHeight)
    this.renderer.autoClear = false
    document.querySelector('.content').appendChild(this.renderer.domElement)

    this.camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 10000)
    this.camera.position.z = 12
    this.camera.frustumCulled = false

    this.scene = new THREE.Scene()
    this.scene.add(this.camera)

    App.holder = new THREE.Object3D()
    App.holder.name = 'holder'
    this.scene.add(App.holder)
    App.holder.sortObjects = false

    App.gui = new dat.GUI()

    this.createManagers(source)

    this.resize()
    window.addEventListener('resize', () => this.resize())
  }

  /**
   * Load audio, detect BPM, then create particles and start the render loop.
   * @param {File|string} source — passed through to AudioManager.loadAudioBuffer()
   */
  async createManagers(source) {
    try {
      App.audioManager = new AudioManager()
      await App.audioManager.loadAudioBuffer(source)
    } catch (err) {
      // Demo track missing or fetch failed — user can upload via the Track button
      console.error('Audio load failed:', err)
      document.querySelector('.user_interaction__label').textContent =
        'Could not load demo track. Upload a file using the Track button.'
      return
    }

    App.bpmManager = new BPMManager()
    App.bpmManager.addEventListener('beat', () => {
      this.particles.onBPMBeat()
    })
    await App.bpmManager.detectBPM(App.audioManager.audio.buffer)

    App.entrainmentManager = new EntrainmentManager()

    document.querySelector('.user_interaction').remove()

    App.audioManager.play()
    this._setupPauseBtn()

    this.particles = new ReativeParticles()
    this.particles.init()

    // If EEG was connected before music started, apply EEG-active defaults
    if (App.eegManager?.isConnected) {
      this.particles.properties.autoMix = false
      this.particles.properties.autoRotate = false
      this.particles.properties.headControl = true
    }
  }

  /** Wire up Jellyfin browser button and create manager + browser instances. */
  _setupJellyfin() {
    const jellyfinManager = new JellyfinManager()
    const jellyfinBrowser = new JellyfinBrowser(jellyfinManager, (url) => {
      if (this.renderer) {
        this._swapAudio(url)
      } else {
        const overlay = document.querySelector('.user_interaction')
        overlay?.removeEventListener('click', this._overlayClickHandler)
        this.init(url)
      }
    })

    document.getElementById('jellyfin-btn').addEventListener('click', () => {
      jellyfinBrowser.show()
    })
  }

  /** Wire up EEG connect/disconnect UI and create the EEGManager instance. */
  _setupEEG() {
    App.eegManager = new EEGManager()

    const controls    = document.getElementById('eeg-controls')
    const btn         = document.getElementById('eeg-connect')
    const batteryEl   = document.getElementById('eeg-status')
    const batteryFill = batteryEl.querySelector('.battery-fill')
    const toggleBtn   = document.getElementById('bio-toggle')
    const panel       = document.getElementById('bio-panel')
    this._hrDisplay   = document.getElementById('heart-rate')
    this._bioHr       = document.getElementById('bio-hr')
    this._qualityDots = document.querySelectorAll('.quality-dot')
    this._bioPanel    = panel
    this._bioToggle   = toggleBtn
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

    toggleBtn.addEventListener('click', () => {
      this._bioVisible = !this._bioVisible
      panel.hidden = !this._bioVisible
      toggleBtn.classList.toggle('active', this._bioVisible)
    })

    App.eegManager.onBatteryLevel = (level) => updateBattery(level)

    App.eegManager.onDisconnected = () => {
      btn.textContent = 'Connect EEG'
      btn.disabled = false
      updateBattery(null)
      toggleBtn.hidden = true
      panel.hidden = true
      this._bioVisible = false
      toggleBtn.classList.remove('active')
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
          toggleBtn.hidden = false
          // Switch to EEG-driven defaults: head control on, auto-mix/rotate off
          if (this.particles) {
            this.particles.properties.autoMix = false
            this.particles.properties.autoRotate = false
            this.particles.properties.headControl = true
          }
          // Init plots on first connect; reset read pointers on subsequent connects
          if (!this._bioDisplay) {
            this._bioDisplay = new BioDataDisplay()
            this._bioDisplay.init()
          } else {
            this._bioDisplay.resetIndices()
          }
        } catch (err) {
          console.error('EEG connect failed:', err)
          btn.textContent = 'Connect EEG'
          btn.disabled = false
        }
      }
    })
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

  /** Swap in a new audio track while the visualizer is running. */
  async _swapAudio(file) {
    if (App.audioManager.isPlaying) {
      App.audioManager.audio.stop()
      App.audioManager.isPlaying = false
    }
    try {
      await App.audioManager.loadAudioBuffer(file)
      await App.bpmManager.detectBPM(App.audioManager.audio.buffer)
      App.audioManager.play()
      if (this._pauseBtn) this._pauseBtn.textContent = '⏸'
    } catch (err) {
      console.error('Audio swap failed:', err)
    }
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

  /** Handle window resize — update camera aspect and renderer size. */
  resize() {
    this.width = window.innerWidth
    this.height = window.innerHeight

    this.camera.aspect = this.width / this.height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(this.width, this.height)
  }

  /** Main render loop — called every animation frame. */
  update() {
    requestAnimationFrame(() => this.update())

    App.eegManager?.update(performance.now())

    if (this._hrDisplay && App.eegManager?.isConnected) {
      const hr = App.eegManager.heartRate
      this._hrDisplay.textContent = hr > 0 ? `♥ ${Math.round(hr)} bpm` : ''
    }

    this._updateBioPanel()

    this.particles?.update(
      App.eegManager?.bandPower,
      App.eegManager?.heartPulse ?? 0,
      App.eegManager?.headPose ?? null,
    )
    App.audioManager?.update()
    App.entrainmentManager?.update(performance.now())

    if (this.renderer) this.renderer.render(this.scene, this.camera)
  }
}
