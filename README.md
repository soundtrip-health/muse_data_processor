# Nouscope

A Muse EEG/PPG/IMU biometric visualizer with optional local-audio playback and real-time neural-entrainment analysis, built with Web Bluetooth and WebGL.

[![Nouscope — Muse EEG/PPG/IMU biometric visualizer](screenshot.png)](https://soundtrip.health/nouscope/)

**[Live demo](https://soundtrip.health/nouscope/)**

## Features

- Live bio-data panel: 4-channel EEG traces, EEG spectrogram (8–50 Hz) and delta/theta panel (0.5–8 Hz), relative band powers (delta, theta, alpha, gamma), PPG heart-rate waveform, and IMU accelerometer/gyroscope traces
- Real-time signal quality per channel
- Multiscale entropy (MSE) EEG complexity readout
- **EEG–music entrainment**: load a local audio file and Nouscope compares the music-tempo tempogram against the EEG tempogram to estimate how strongly the brain entrains to the beat
- PPG heart-rate detection (MSPTDfast)
- IMU head-pose estimation
- Full-screen data view and JSONL session recording (raw EEG/PPG/IMU + derived metrics)

## Browser Support

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| Bio-data panel / audio | ✅ | ✅ | ✅ | ✅ |
| EEG (Web Bluetooth) | ✅ | ✅ | ❌ | ❌ |

**Note:** EEG/Bluetooth features require Chrome or Edge. HTTPS is required for Web Bluetooth in production.

## Getting Started

### Prerequisites

- Node.js 18+
- npm

### Install & Run

```bash
npm install
npm run dev
```

Open `http://localhost:5173` (any modern browser; use Chrome or Edge to develop or test Web Bluetooth / EEG).

## Usage

1. Click **Connect EEG** to pair a Muse headset via Bluetooth.
2. Click **◉** to open the live bio-data panel (or **⛶** for the full-screen data view).
3. Optionally click **↑ Track** to load a local audio file. Playback enables the **AUDIO TEMPO** panel and the EEG–music entrainment meter. Use **⏸** to pause/resume.
4. Click **⏺** to record the session to a timestamped `.jsonl` file.

The app works with EEG only (no audio) or with audio only (the tempogram is shown but the entrainment index needs both an EEG signal and playing audio).

## EEG Integration

Requires a [Muse](https://choosemuse.com/) EEG headset (Muse 2 or Muse S) and Chrome or Edge.

| EEG band | Frequency (approx.) | Mental state |
|----------|---------------------|--------------|
| Delta (1–4 Hz) | Deep sleep | Excluded from the relative-power panel by default (movement-prone; would swamp higher bands) |
| Theta (4–8 Hz) | Drowsy / relaxed | |
| Alpha (8–13 Hz) | Calm / idle | |
| Beta (13–30 Hz) | Focused / alert | |
| Gamma (30–50 Hz) | High cognition | |

**PPG / Heart Rate** — detects heartbeats from the Muse's infrared sensor.

**IMU / Head Pose** — accelerometer pitch and roll are estimated and plotted alongside the raw gyroscope traces.

## Developer Guide

For a detailed explanation of the signal-processing algorithms (EEG band powers, PPG detection, entrainment index, multiscale entropy), see [`docs/algorithms.md`](docs/algorithms.md).

## Architecture

```
src/js/
├── index.js                    — entry point, instantiates App
├── App.js                      — managers, UI wiring, update loop
├── managers/
│   ├── AudioManager.js         — audio loading + spectral-flux novelty for entrainment
│   ├── BPMManager.js           — tempo detection (bpmValue for recording metadata)
│   ├── EEGManager.js           — Muse BT, EEG bands, PPG heart rate, IMU head pose
│   ├── EntrainmentManager.js   — audio/EEG tempograms + entrainment index
│   ├── ComplexityManager.js    — multiscale entropy (MSE) on EEG
│   └── RecordingManager.js     — JSONL session recorder
└── ui/
    └── BioDataDisplay.js       — live EEG / PPG / IMU / spectrogram / entrainment panel
```

There is no 3D scene — the bio-data panel (webgl-plot line traces + 2D canvas heatmaps) is the visualization.

## Credits

- EEG/PPG/IMU integration: [Soundtrip](https://github.com/soundtrip-health)
- [muse-js](https://github.com/soundtrip-health/muse-js) — Web Bluetooth Muse SDK
- [web-audio-beat-detector](https://github.com/chrisguttandin/web-audio-beat-detector) — BPM detection
- [webgl-plot](https://github.com/danchitnis/webgl-plot) — line plotting
- [Three.js](https://threejs.org) — Web Audio helpers (AudioListener / AudioAnalyser)

## License

MIT — see [LICENSE](LICENSE)
