import { ColorRGBA } from 'webgl-plot'

/**
 * palette — bridge between the CSS design tokens and the WebGL data traces.
 *
 * Colors are defined once, in `src/scss/includes/_tokens.scss`, as CSS custom
 * properties. This module reads those same properties at runtime and converts
 * them to `ColorRGBA` (0–255) so the plots, the HTML legends, and the
 * stylesheet can never drift apart. See docs/design-system.md.
 *
 * Read lazily (inside BioDataDisplay.init(), which runs on EEG connect), never
 * at module load — the stylesheet must be parsed before getComputedStyle can
 * resolve the custom properties.
 */

/** Read a CSS custom property (e.g. '--eeg-tp9') off :root as a trimmed string. */
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/** Parse '#rgb' or '#rrggbb' → [r, g, b] (0–255). */
export function hexToRgb(hex) {
  let h = hex.replace('#', '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  const n = parseInt(h, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** [r, g, b] tuple for a color token (for colormap / interpolation math). */
export function rgbVar(name) {
  return hexToRgb(cssVar(name))
}

/** ColorRGBA (alpha 255) for a single color token. */
export function colorVar(name) {
  const [r, g, b] = rgbVar(name)
  return new ColorRGBA(r, g, b, 255)
}

/** ColorRGBA[] for a list of color tokens, in order. */
export function colorVars(names) {
  return names.map(colorVar)
}
