import * as THREE from 'three'
import gsap from 'gsap'
import vertex from './glsl/vertex.glsl'
import fragment from './glsl/fragment.glsl'
import App from '../App'

// --- Constants ---
const BASE_SIZE = 1.1
const BASE_MAX_DISTANCE = 1.8
const BEAT_ROTATE_CHANCE = 0.3
const BEAT_RESET_CHANCE = 0.3
const CAMERA_Z = 10          // nominal camera depth (for z-position tweens)
const CAMERA_Z_RANGE = 1     // ±range for random z placement
const EEG_LERP_RATE = 0.06  // per-frame lerp toward latest EEG bands (~60 fps → ~4 Hz convergence)

/**
 * Bio feature sources available for mapping to viz parameters.
 * EEG band values are per-band activity deviations above the adaptive 1/f baseline:
 *   0   = band power at or below the recent typical (aperiodic) level
 *   1.0 = band is twice its recent typical level (strong oscillatory elevation)
 * 'hr' is the heart-rate phase oscillator (0–1, cubed-sine shape), unchanged.
 * 'none' produces zero contribution.
 */
const BIO_SOURCES = ['none', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'hr', 'entrain']

/**
 * Per viz-parameter bio mapping ranges.
 *   min     — weight slider left edge  (no contribution)
 *   max     — weight slider right edge (maximum contribution)
 *   default — starting value
 *
 * Most parameters use **multiplicative** scaling:
 *   uniform *= (1 + sources[source] * weight)
 * This makes EEG modulate the audio reactivity — a focused brain amplifies
 * the music's effect, a relaxed brain softens it.
 *
 * Exceptions (direct assignment):
 *   heartPulse — uniform = sources[source] * weight  (standalone color flush)
 *   hueShift   — uniform = sources[source] * weight  (hue rotation amount)
 */
const BIO_RANGE = {
  amplitude:   { min: 0.0, max: 1.0, default: 0.5  },
  offsetGain:  { min: 0.0, max: 2.0, default: 1.0  },
  size:        { min: 0.0, max: 3.0, default: 1.5  },
  maxDistance:  { min: 0.0, max: 2.0, default: 1.0  },
  frequency:   { min: 0.0, max: 3.0, default: 1.5  },
  hueShift:    { min: 0.0, max: 0.25, default: 0.12 },
  heartPulse:  { min: 0.0, max: 2.0, default: 1.0  },
}

/**
 * Per-band audio gain ranges.
 *   min     — slider left  (silence the band's contribution)
 *   max     — slider right (2× the default contribution)
 *   default — mid/high: 1.0 = prior tuning; bass: 0.5 = calmer default speed
 *
 * bass → animation speed (time increment)
 * mid  → turbulence (offsetGain baseline)
 * high → displacement amplitude baseline
 */
const AUDIO_RANGE = {
  bass: { min: 0.0, max: 2.0, default: 0.5 },
  mid:  { min: 0.0, max: 2.0, default: 1.0 },
  high: { min: 0.0, max: 2.0, default: 1.0 },
}

/**
 * ReactiveParticles
 *
 * A Three.js Object3D that renders an audio-reactive particle system.
 * Each frame, audio frequency data and optional EEG band powers are mapped
 * to ShaderMaterial uniforms controlling particle size, displacement, and color.
 *
 * On each BPM beat (when Auto Mix / Auto Rotate are on), the mesh may be
 * rebuilt as a new randomized cylinder and/or holder rotation tweened via GSAP.
 *
 * Optional: IMU head-pose (pitch/roll) overrides auto-rotate when headControl is enabled.
 */
export default class ReactiveParticles extends THREE.Object3D {
  constructor() {
    super()
    this.name = 'ReactiveParticles'
    this.time = 0
    this.properties = {
      startColor: 0xff00ff,
      endColor: 0x00ffff,
      autoMix: true,
      autoRotate: true,
      headControl: false,
      imuStrength: 1.0,
    }

    // Per-band audio gain: 1.0 = default behavior; slider range defined in AUDIO_RANGE
    this.audioGains = {
      bass: AUDIO_RANGE.bass.default,
      mid:  AUDIO_RANGE.mid.default,
      high: AUDIO_RANGE.high.default,
    }

    // Bio→viz mapping: source selects the input signal; weight scales its contribution.
    // Most use multiplicative scaling (EEG amplifies audio reactivity).
    this.bioMapping = {
      amplitude:   { source: 'gamma', weight: BIO_RANGE.amplitude.default },
      offsetGain:  { source: 'beta',  weight: BIO_RANGE.offsetGain.default },
      size:        { source: 'theta', weight: BIO_RANGE.size.default },
      maxDistance:  { source: 'alpha', weight: BIO_RANGE.maxDistance.default },
      frequency:   { source: 'beta',  weight: BIO_RANGE.frequency.default },
      hueShift:    { source: 'gamma', weight: BIO_RANGE.hueShift.default },
      heartPulse:  { source: 'hr',    weight: BIO_RANGE.heartPulse.default },
    }

    // Base curl-field frequency; GSAP tweens this on beat, update() applies EEG multiplier
    this._baseFrequency = 2

    // Per-frame smoothed EEG bands — lerped toward latest bandPower each frame
    this._smoothedBands = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
  }

  /** Attach to the scene holder, create ShaderMaterial, build initial mesh, add GUI. */
  init() {
    App.holder.add(this)

    this.holderObjects = new THREE.Object3D()
    this.add(this.holderObjects)

    this.material = new THREE.ShaderMaterial({
      side: THREE.DoubleSide,
      vertexShader: vertex,
      fragmentShader: fragment,
      transparent: true,
      uniforms: {
        time:        { value: 0 },
        offsetSize:  { value: 2 },
        size:        { value: BASE_SIZE },
        frequency:   { value: 2 },
        amplitude:   { value: 1 },
        offsetGain:  { value: 0 },
        maxDistance: { value: BASE_MAX_DISTANCE },
        startColor:  { value: new THREE.Color(this.properties.startColor) },
        endColor:    { value: new THREE.Color(this.properties.endColor) },
        heartPulse:  { value: 0 },
        hueShift:    { value: 0 },
      },
    })

    this.addGUI()
    this.resetMesh()
  }

  /** Create a randomized box-geometry particle mesh and tween it into view. */
  createBoxMesh() {
    const widthSeg  = Math.floor(THREE.MathUtils.randInt(5, 20))
    const heightSeg = Math.floor(THREE.MathUtils.randInt(1, 40))
    const depthSeg  = Math.floor(THREE.MathUtils.randInt(5, 80))
    this.geometry = new THREE.BoxGeometry(1, 1, 1, widthSeg, heightSeg, depthSeg)

    this.material.uniforms.offsetSize.value = Math.floor(THREE.MathUtils.randInt(30, 60))
    this.material.needsUpdate = true

    this.pointsMesh = new THREE.Object3D()
    this.pointsMesh.rotateX(Math.PI / 2)
    this.holderObjects.add(this.pointsMesh)

    const pointsMesh = new THREE.Points(this.geometry, this.material)
    this.pointsMesh.add(pointsMesh)

    gsap.to(this.pointsMesh.rotation, {
      duration: 3,
      x: Math.random() * Math.PI,
      z: Math.random() * Math.PI * 2,
      ease: 'none',
    })

    gsap.to(this.position, {
      duration: 0.6,
      z: THREE.MathUtils.randInt(CAMERA_Z - CAMERA_Z_RANGE, CAMERA_Z + CAMERA_Z_RANGE),
      ease: 'elastic.out(0.8)',
    })
  }

  /** Create a randomized cylinder-geometry particle mesh and tween it into view. */
  createCylinderMesh() {
    const radialSeg = Math.floor(THREE.MathUtils.randInt(1, 3))
    const heightSeg = Math.floor(THREE.MathUtils.randInt(1, 5))
    this.geometry = new THREE.CylinderGeometry(1, 1, 4, 64 * radialSeg, 64 * heightSeg, true)

    this.material.uniforms.offsetSize.value = Math.floor(THREE.MathUtils.randInt(30, 60))
    this.material.uniforms.size.value = 2
    this.material.needsUpdate = true

    this.pointsMesh = new THREE.Points(this.geometry, this.material)
    this.pointsMesh.rotation.set(Math.PI / 2, 0, 0)
    this.holderObjects.add(this.pointsMesh)

    let rotY = 0
    let posZ = THREE.MathUtils.randInt(CAMERA_Z - CAMERA_Z_RANGE, CAMERA_Z + CAMERA_Z_RANGE)

    if (Math.random() < 0.2) {
      rotY = Math.PI / 2
      posZ = THREE.MathUtils.randInt(10, 11.5)
    }

    gsap.to(this.holderObjects.rotation, {
      duration: 0.2,
      y: rotY,
      ease: 'elastic.out(0.2)',
    })

    gsap.to(this.position, {
      duration: 0.6,
      z: posZ,
      ease: 'elastic.out(0.8)',
    })
  }

  /**
   * Called on each BPM beat. Randomly triggers an auto-rotate tween and/or
   * a geometry reset. Skips auto-rotate when head-control (IMU) is active.
   */
  onBPMBeat() {
    const duration = App.bpmManager.getBPMDuration() / 1000

    if (App.audioManager.isPlaying) {
      if (Math.random() < BEAT_ROTATE_CHANCE && this.properties.autoRotate && !this.properties.headControl) {
        gsap.to(this.holderObjects.rotation, {
          duration: Math.random() < 0.8 ? 15 : duration,
          z: Math.random() * Math.PI,
          ease: 'elastic.out(0.2)',
        })
      }

      if (Math.random() < BEAT_RESET_CHANCE) {
        this.resetMesh()
      }
    }
  }

  /** Replace the current mesh with a new cylinder if autoMix is enabled. */
  resetMesh() {
    if (this.properties.autoMix) {
      this.destroyMesh()
      this.createCylinderMesh()

      gsap.to(this, {
        duration: App.bpmManager ? (App.bpmManager.getBPMDuration() / 1000) * 2 : 2,
        _baseFrequency: THREE.MathUtils.randFloat(0.5, 3),
        ease: 'expo.easeInOut',
      })
    }
  }

  /** Remove the current points mesh and dispose its GPU resources. */
  destroyMesh() {
    if (this.pointsMesh) {
      gsap.killTweensOf(this.pointsMesh)
      gsap.killTweensOf(this.pointsMesh.rotation)
      this.holderObjects.remove(this.pointsMesh)
      this.pointsMesh.geometry?.dispose()
      this.pointsMesh.material?.dispose()
      this.pointsMesh = null
    }
  }

  /**
   * Update shader uniforms from audio and EEG data. Called every frame.
   * @param {object|null} eegBands  — { delta, theta, alpha, beta, gamma } activity deviations ≥ 0, or null
   * @param {number}      heartPulse — heart-rate oscillator value (0–1)
   * @param {object|null} headPose  — { pitch, roll } in radians, or null
   */
  update(eegBands = null, heartPulse = 0, headPose = null) {
    // Per-frame lerp toward latest EEG values for smooth sub-update interpolation
    for (const band of Object.keys(this._smoothedBands)) {
      const target = eegBands?.[band] ?? 0
      this._smoothedBands[band] += EEG_LERP_RATE * (target - this._smoothedBands[band])
    }

    const sources = {
      none:    0,
      delta:   this._smoothedBands.delta,
      theta:   this._smoothedBands.theta,
      alpha:   this._smoothedBands.alpha,
      beta:    this._smoothedBands.beta,
      gamma:   this._smoothedBands.gamma,
      hr:      heartPulse,
      entrain: App.entrainmentManager?.entrainment ?? 0,
    }

    let amplitude, offsetGain, size, maxDistance

    if (App.audioManager?.isPlaying) {
      // Audio baseline — audioGains scale each band's contribution
      amplitude   = 0.8 + THREE.MathUtils.mapLinear(App.audioManager.frequencyData.high, 0, 0.6, -0.1, 0.2) * this.audioGains.high
      offsetGain  = App.audioManager.frequencyData.mid * 0.6 * this.audioGains.mid
      size        = BASE_SIZE
      maxDistance  = BASE_MAX_DISTANCE

      // Bass gain scales animation speed (floor keeps time advancing at very low gain)
      const t = THREE.MathUtils.mapLinear(App.audioManager.frequencyData.low, 0.6, 1, 0.2, 0.5)
      this.time += Math.max(0.01, THREE.MathUtils.clamp(t, 0.2, 0.5) * this.audioGains.bass)
    } else {
      amplitude   = 1
      offsetGain  = 0
      size        = BASE_SIZE
      maxDistance  = BASE_MAX_DISTANCE
      this._baseFrequency = 0.8
      this.time += 0.2
    }

    // EEG multiplicative modulation — brain state scales how strongly audio
    // (or idle defaults) affect the viz. Focused brain amplifies, relaxed softens.
    amplitude  *= (1 + sources[this.bioMapping.amplitude.source]  * this.bioMapping.amplitude.weight)
    offsetGain *= (1 + sources[this.bioMapping.offsetGain.source] * this.bioMapping.offsetGain.weight)
    size       *= (1 + sources[this.bioMapping.size.source]       * this.bioMapping.size.weight)
    maxDistance *= (1 + sources[this.bioMapping.maxDistance.source] * this.bioMapping.maxDistance.weight)

    this.material.uniforms.amplitude.value  = amplitude
    this.material.uniforms.offsetGain.value = offsetGain
    this.material.uniforms.size.value       = size
    this.material.uniforms.maxDistance.value = maxDistance

    // Curl field frequency: base from GSAP beat tweens, modulated by EEG
    this.material.uniforms.frequency.value = this._baseFrequency *
      (1 + sources[this.bioMapping.frequency.source] * this.bioMapping.frequency.weight)

    // Hue shift — rotates the color palette based on brain state
    this.material.uniforms.hueShift.value =
      sources[this.bioMapping.hueShift.source] * this.bioMapping.hueShift.weight

    // Color flush — driven by bio mapping, applied regardless of audio state
    this.material.uniforms.heartPulse.value =
      sources[this.bioMapping.heartPulse.source] * this.bioMapping.heartPulse.weight

    // Head-control mode: map IMU pitch/roll directly onto geometry orientation,
    // overriding any GSAP auto-rotate tweens
    if (this.properties.headControl && headPose) {
      gsap.killTweensOf(this.holderObjects.rotation)
      const imuStr = this.properties.imuStrength * 0.8
      this.holderObjects.rotation.x = headPose.pitch * imuStr
      this.holderObjects.rotation.y = headPose.roll  * imuStr
    }

    this.material.uniforms.time.value = this.time
  }

  addGUI() {
    const gui = App.gui

    const particlesFolder = gui.addFolder('PARTICLES')
    particlesFolder
      .addColor(this.properties, 'startColor')
      .listen()
      .name('Start Color')
      .onChange((e) => {
        this.material.uniforms.startColor.value = new THREE.Color(e)
      })
    particlesFolder
      .addColor(this.properties, 'endColor')
      .listen()
      .name('End Color')
      .onChange((e) => {
        this.material.uniforms.endColor.value = new THREE.Color(e)
      })

    const visualizerFolder = gui.addFolder('VISUALIZER')
    visualizerFolder.add(this.properties, 'autoMix').listen().name('Auto Mix')
    visualizerFolder.add(this.properties, 'autoRotate').listen().name('Auto Rotate')
    visualizerFolder
      .add(this.properties, 'headControl')
      .listen()
      .name('Head Control (IMU)')
      .onChange((enabled) => {
        if (!enabled) {
          gsap.to(this.holderObjects.rotation, { duration: 1.5, x: 0, y: 0, ease: 'power2.out' })
        }
      })
    visualizerFolder.add(this.properties, 'imuStrength', 0, 3).name('IMU Strength')
    const buttonShowCylinder = {
      showCylinder: () => {
        this.destroyMesh()
        this.createCylinderMesh()
        this.properties.autoMix = false
      },
    }
    visualizerFolder.add(buttonShowCylinder, 'showCylinder').name('Reset Cylinder')

    const audioFolder = gui.addFolder('AUDIO')
    audioFolder.add(this.audioGains, 'bass', AUDIO_RANGE.bass.min, AUDIO_RANGE.bass.max).name('Bass Gain')
    audioFolder.add(this.audioGains, 'mid',  AUDIO_RANGE.mid.min,  AUDIO_RANGE.mid.max).name('Mid Gain')
    audioFolder.add(this.audioGains, 'high', AUDIO_RANGE.high.min, AUDIO_RANGE.high.max).name('High Gain')

    // MAPPING — one sub-folder per viz parameter with source dropdown + weight slider
    const mappingFolder = gui.addFolder('MAPPING')
    const paramLabels = {
      amplitude:   'Amplitude',
      offsetGain:  'Turbulence',
      size:        'Particle Size',
      maxDistance:  'Spread Radius',
      frequency:   'Field Chaos',
      hueShift:    'Hue Shift',
      heartPulse:  'Color Flush',
    }
    for (const [param, label] of Object.entries(paramLabels)) {
      const sub = mappingFolder.addFolder(label)
      sub.add(this.bioMapping[param], 'source', BIO_SOURCES).name('Source')
        .onChange(() => this._syncNormalizeBands())
      sub.add(this.bioMapping[param], 'weight', BIO_RANGE[param].min, BIO_RANGE[param].max).name('Weight')
    }

    // Push initial active-band set so EEGManager normalises correctly from the start
    this._syncNormalizeBands()
  }

  /**
   * Derive which EEG bands are currently mapped to at least one viz parameter and
   * push the resulting Set to EEGManager so unmapped bands are excluded from the
   * relative-power normalisation (prevents unmapped high-power bands like delta
   * from consuming most of the normalised share).
   */
  _syncNormalizeBands() {
    const EEG_BANDS = new Set(['delta', 'theta', 'alpha', 'beta', 'gamma'])
    const active = new Set(
      Object.values(this.bioMapping)
        .map(m => m.source)
        .filter(s => EEG_BANDS.has(s))
    )
    // If nothing is mapped (all 'none'), fall back to full set to avoid all-zeros output
    if (App.eegManager) {
      App.eegManager.normalizeBands = active.size > 0 ? active : new Set(EEG_BANDS)
    }
  }
}
