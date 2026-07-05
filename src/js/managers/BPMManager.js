import { guess } from 'web-audio-beat-detector'
import App from '../App'

// --- Constants ---
const DEFAULT_BPM = 120

/**
 * BPMManager
 *
 * Detects the tempo (BPM) of a decoded AudioBuffer using
 * `web-audio-beat-detector`. The detected value is exposed as `bpmValue`
 * (used in recording metadata / the `music` record) — no beat events are
 * dispatched. Falls back to 120 BPM if detection fails.
 *
 * @example
 *   const mgr = new BPMManager()
 *   await mgr.detectBPM(audioBuffer)
 *   console.log(mgr.bpmValue)
 */
export default class BPMManager {
  constructor() {
    this.bpmValue = 0
  }

  /**
   * Store a new BPM value and log it to any active recording.
   * @param {number} bpm
   */
  setBPM(bpm) {
    this.bpmValue = bpm
    App.recordingManager?.recordMusicTempo(bpm)
  }

  /**
   * Analyze an AudioBuffer to detect BPM.
   * Falls back to 120 BPM if detection throws.
   * @param {AudioBuffer} audioBuffer
   * @returns {Promise<void>}
   */
  async detectBPM(audioBuffer) {
    try {
      const { bpm } = await guess(audioBuffer)
      this.setBPM(bpm)
    } catch {
      this.setBPM(DEFAULT_BPM) // fallback if detection fails
    }
  }
}
