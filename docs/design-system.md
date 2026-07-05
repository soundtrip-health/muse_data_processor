# Nouscope — Design System

The current visual language of Nouscope, extracted from the shipping code
(`src/scss/includes/base.scss`, `src/js/ui/BioDataDisplay.js`, `index.html`).
Use this as the ground truth when proposing UI changes — it describes *where
things stand today*, not an aspiration.

**Implementation:** every color below is a **CSS custom property** defined once
in [`src/scss/includes/_tokens.scss`](../src/scss/includes/_tokens.scss) — the
single source of truth. The stylesheet consumes them via `var(--…)`; the WebGL
data traces read the *same* properties at runtime through
[`src/js/ui/palette.js`](../src/js/ui/palette.js), so the plots, the HTML
legends, and the CSS can never drift apart. Change a color in one place.

---

## 1. Product character

Nouscope is a **real-time neuro-instrument**, not a dashboard. It reads a Muse
headset (EEG / PPG / IMU) and paints the signal as it arrives — scrolling
line traces and viridis heatmaps on a pure-black field. The data *is* the
interface: chrome is deliberately minimal, pushed to a single control bar in
the bottom-left, so nothing competes with the living signal.

Design principles that follow from that:

- **Black is the canvas, not a background.** The signal glows against true
  black (`#000`). Panels are barely-there dark scrims, not cards.
- **Color is data, not decoration.** Nearly every hue in the app is a
  *channel identity* (which electrode, which band, which sensor). Don't spend
  those colors on chrome — chrome is monochrome white-on-black at low opacity.
- **One committed theme.** This is a dark instrument. There is no light mode
  and shouldn't be — a clinical/observatory darkroom aesthetic is the point.
- **Legibility at a glance while moving.** Readouts use tabular figures,
  uppercase micro-labels, and generous letter-spacing so values can be read
  peripherally during a session.

---

## 2. Neutrals & surfaces

Chosen, not defaulted: the text white carries a **warm bias** (`#FFFBF5`)
against pure black, giving the darkroom a slightly analog, phosphor feel
rather than clinical blue-white.

| Token | Value | Use |
|-------|-------|-----|
| `--color-bg` | `#000000` | Page ground. True black. |
| `--color-text` | `#FFFBF5` | Warm white. Primary text & icons. |
| Panel fill | `rgba(0,0,0,0.75)` | Bio-panel scrim over content. |
| Hairline border | `rgba(255,255,255,0.12)` | Panel edge. |
| Control border | `rgba(255,255,255,0.40)` | Primary buttons (Connect, Track, mic). |
| Button border | `rgba(255,255,255,0.25)` | Secondary buttons (record). |
| Battery stroke | `rgba(255,255,255,0.30)` | Idle iconography. |
| Hover wash | `rgba(255,255,255,0.10)` | Button hover fill. |

**Text opacity ramp** (all white over black) — hierarchy is carried by
opacity, not size alone:

| Opacity | Role |
|---------|------|
| `1.0` (`#FFFBF5`) | Primary button labels, active values |
| `0.70` | Heart-rate readout |
| `0.55` | Band / MSE numeric values |
| `0.50` | Bio-panel HR, secondary values |
| `0.45` | Section micro-labels (`EEG`, `SPECTROGRAM`…) |
| `0.30` | Axis tick numbers |

---

## 3. Typography

- **Display / UI face:** **Degular** (Adobe Typekit, kit `rsf5rhz`), weight
  400, loaded via `@import url('https://use.typekit.net/rsf5rhz.css')`.
  Fallback stack: `'degular', system-ui, -apple-system, sans-serif`.
  > Note for artifacts: Typekit is CDN-hosted and blocked by the artifact CSP.
  > When mocking outside the app, substitute a geometric/humanist sans
  > (e.g. system-ui) and label it as a stand-in — production is Degular.
- **Root size:** `16px`. All sizes below are `rem`.
- **Numerics:** `font-variant-numeric: tabular-nums` on *every* readout
  (HR, band values, MSE, record timer) so digits don't jitter as they update.
- **Micro-labels:** uppercase, `letter-spacing: 0.06–0.08em`, opacity 0.45.
  These name each data section.

### Type scale (live)

| rem | px | Role |
|-----|----|------|
| 0.50 | 8  | Heatmap axis ticks |
| 0.60 | 9.6 | Band / MSE / entrain values |
| 0.65 | 10.4 | Section labels, legends, HR-in-panel |
| 0.70 | 11.2 | Record timer |
| 0.75 | 12  | Controls: Connect EEG, ↑ Track, heart-rate |
| 0.85 | 13.6 | Pause / mic buttons |
| 0.90 | 14.4 | Upload-btn (legacy) |

### Full-screen data view (default when connected)

The full-screen grid bumps the scale up for legibility across the room:

| Element | Normal | Full-screen |
|---------|--------|-------------|
| Section label | 0.65 | **0.90** |
| Band / MSE legend | 0.65 | **0.90** |
| Band / MSE value | 0.60 | **0.80** |
| HR / entrain / MSE value / IMU legend | 0.60–0.65 | **0.85** |
| Axis ticks | 0.50 | **0.70** |

---

## 4. Data-channel color system

This is the heart of the palette. Each color is an **identity** carried
consistently between the trace, its legend glyph, and its numeric readout.
All defined as `--eeg-*`, `--band-*`, `--ppg`, `--imu-*`, `--mse-*` tokens;
`BioDataDisplay.js` builds its `ColorRGBA` line colors from them via
`palette.js`, and the `index.html` legend glyphs use `.glyph-*` classes that
point at the same tokens.

### EEG electrodes (4-channel stacked trace)

| Channel | Hex | RGB | Name |
|---------|-----|-----|------|
| TP9  | `#4DD9FF` | 77, 217, 255 | cyan |
| AF7  | `#66FF80` | 102, 255, 128 | green |
| AF8  | `#FFA633` | 255, 166, 51 | orange |
| TP10 | `#E566FF` | 229, 102, 255 | magenta |

### EEG frequency bands (δ θ α β γ)

Ordered cool→warm to read as an ascending frequency spectrum.

| Band | Hex | RGB | Glyph |
|------|-----|-----|-------|
| δ delta | `#A78BFA` | 167, 139, 250 | violet |
| θ theta | `#60A5FA` | 96, 165, 250 | blue |
| α alpha | `#34D399` | 52, 211, 153 | green |
| β beta  | `#FBBF24` | 251, 191, 36 | amber |
| γ gamma | `#F87171` | 248, 113, 113 | red |

### PPG (heart)

| | Hex | RGB |
|-|-----|-----|
| PPG trace | `#FF8080` | 255, 128, 128 (salmon) |

### IMU (6-line: accel = blues, gyro = reds)

| Axis | Accel (blue) | Gyro (red) |
|------|-------------|------------|
| x | `#4DB2FF` | `#FF6666` |
| y | `#3380E6` | `#D94D4D` |
| z | `#80D2FF` | `#FF9980` |

Legend collapses to two swatches: **A** (`#4FC3F7`) / **G** (`#EF9A9A`).

### Complexity (MSE) — 5-scale gradient

Interpolated violet→amber across scales τ=1…9, tying it to the band spectrum's
endpoints:

`#A78BFA` (τ=1) → … → `#FBBF24` (τ=9)

---

## 5. Heatmap colormap — Viridis

All three heatmaps (main spectrogram 8–50 Hz, delta/theta 0.5–8 Hz, audio
tempo 0.5–5 Hz) map `log₁₀(power)` through a **256-entry Viridis LUT** with
robust percentile auto-scaling. Viridis is perceptually uniform and
colorblind-safe — keep it; it's the visual signature of the whole app.

9 key stops (t, R, G, B):

```
0.000  68,  1, 84     (deep violet)
0.250  59, 82, 139
0.500  33, 145, 140   (teal)
0.750 122, 209, 81
1.000 253, 231, 37    (yellow)
```

Heatmaps render as 2px pixel columns with `image-rendering: pixelated` to keep
frequency bins crisp under CSS scaling.

---

## 6. Semantic / status colors

Distinct from the data palette — reserved for system state.

| State | Hex | Where |
|-------|-----|-------|
| Good / connected | `#00E676` | Signal-quality dot (good), battery (good) |
| Marginal (dot) | `#FFEE58` | Signal-quality dot |
| Battery warn | `#FFB300` | Battery 20–49% |
| Poor / error (dot) | `#EF5350` | Signal-quality dot, record-active |
| Battery low | `#FF5252` | Battery <20% |
| Recording | `#EF5350` | Record button active (pulse) |
| Mic live | `#EF4444` | Mic button active |

**Entrainment meter** uses a gradient (green→blue) as a *scalar* accent, not a
status: `linear-gradient(90deg, #34D399, #60A5FA)`.

Status colors get a matching `box-shadow` glow (`0 0 4px <color>`) on the
signal dots so state reads at a glance.

---

## 7. Spacing & layout

Spacing is a small `rem` set — keep to it:

`0.25 · 0.3 · 0.4 · 0.5 · 0.6 · 0.8 · 1 · 1.5`

| Context | Value |
|---------|-------|
| Panel padding (normal) | `0.6rem` |
| Panel padding (full-screen) | `1.5rem` |
| Section gap (panel) | `0.5rem` |
| Grid gap (full-screen) | `1rem` |
| Control-bar gap | `0.6rem` |
| Screen inset (controls) | `1.5rem` from bottom-left |

### Full-screen data view (the default layout)

When EEG connects, the app auto-enters `body.fullscreen-bio`:

- `#bio-panel` → `position: fixed; inset: 0`, 2-column CSS grid
  (`grid-template-columns: 1fr 1fr`), `grid-auto-rows: minmax(0, 1fr)`,
  `gap: 1rem`, `padding: 1.5rem`.
- **EEG section spans the full width** (`grid-column: 1 / -1`) as the hero row;
  remaining sections fill the two columns.
- Canvases scale to their grid cell (`width/height: 100%`); heatmaps stay
  pixel-crisp, line traces scale smoothly.
- The control bar (`#eeg-controls`) stays fixed bottom-left, `z-index` above
  the panel, always clickable.

There are **no toggle buttons** for the panel or full-screen — the data view is
the default and appears on connect, clears on disconnect.

---

## 8. Components

All controls share the same recipe: transparent fill, hairline white border,
white text, subtle hover (border brightens + `rgba(255,255,255,0.1)` wash).
No rounded corners, no drop shadows — flat, instrument-panel switches.

| Component | Border | Font | State color |
|-----------|--------|------|-------------|
| `Connect EEG` / `↑ Track` | `0.40` white | 0.75rem, `0.05em` | — |
| Pause / mic | `0.40` white | 0.85rem | mic active → `#EF4444` |
| Record `⏺` | `0.25` white | 0.75rem | active → `#EF5350` + pulse |
| Battery (SVG) | stroke `0.30` | — | good/warn/low semantic |
| Quality dots | — | 7px circle | good/marginal/poor + glow |

**Record pulse:** `@keyframes record-pulse` — expanding `box-shadow` ring in
`rgba(239,83,80,·)` over 1.4s, ease-in-out, infinite.

**Transitions:** border/background/color at `0.2s`; battery fill/stroke at
`0.3–0.5s`; quality dots at `0.4s`; entrainment bar width at `0.3s ease-out`.

---

## 9. Voice

Labels are terse, technical, uppercase: `EEG`, `SPECTROGRAM`, `DELTA / THETA`,
`AUDIO TEMPO`, `EEG BANDS`, `COMPLEXITY (MSE)`, `PPG`, `IMU`. Values are bare
numbers with tight units (`82 bpm`, `0.29`, `τ=1`). Buttons state the action
(`Connect EEG` / `Disconnect EEG`). No marketing tone — this reads like lab
equipment, and should.

---

## 10. When designing new surfaces

- Reach for the **data palette** only when a color *means* a channel; otherwise
  stay monochrome white-on-black via the opacity ramp.
- New status? Extend §6, don't invent a new hue family.
- Keep viridis for anything heat/intensity-mapped — consistency is the brand.
- Respect the black. New panels are dark scrims, not light cards.
- Match the type scale and tabular-nums; don't introduce a second typeface.
