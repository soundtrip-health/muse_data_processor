# muse_data_processor

A small C++ command-line tool that turns a raw Muse EEG stream into a real-time
text time-series of four biomarkers. It reads a recorded Muse session today
(`.jsonl`) and is structured for a live BLE source later — the processing is
fully streaming and causal.

Each output row is:

```
t_seconds  entropy  theta_alpha  alpha_symmetry  quality
```

| Column | Meaning |
|---|---|
| `t` | seconds since the start of capture (window end) |
| `entropy` | multiscale-entropy (MSE) complexity — mean sample entropy over scales {1,3,5,7,9} of the quality-weighted 4-channel average (8 s window) |
| `theta_alpha` | quality-weighted raw band-power ratio, theta (6 Hz) / alpha (10 Hz), 1 s window |
| `alpha_symmetry` | frontal alpha asymmetry, `ln(alpha_AF8) − ln(alpha_AF7)` (right − left) |
| `quality` | one blended data-quality index in [0,1]: `0.5·(good_channels/4) + 0.5·(1 − min(best_rms,100)/100)` |

Muse electrode order is fixed: `0=TP9, 1=AF7, 2=AF8, 3=TP10`. Values are `nan`
until enough data has accumulated (theta/alpha & symmetry after ~1 s, entropy
after ~8 s) or when a channel window contains an unfillable dropout.

## Build

Requires a C++17 compiler (`g++`). No CMake, no external libraries.

```bash
cd cpp && make          # produces cpp/museproc
```

## Run

```bash
./cpp/museproc data/session1.jsonl > metrics.txt         # batch
./cpp/museproc data/session1.jsonl --realtime            # pace to wall-clock
cat data/session1.jsonl | ./cpp/museproc -               # stream from stdin
```

Options:

| Flag | Default | Effect |
|---|---|---|
| `--hop N` | 128 | samples between output rows (128 ≈ 2 Hz at 256 Hz) |
| `--csv` | off | comma-separated output instead of space |
| `--realtime` | off | sleep so rows emit at wall-clock time (emulates a live stream) |
| `--no-notch` | on | disable the causal 60 Hz mains notch |
| `--line-hz F` | 60 | mains frequency for the notch |
| `--fs F` | 256 | EEG sample rate |

## Plot (test harness)

`harness/plot_metrics.py` draws a 4-panel figure from the tool's output. It is a
self-contained [`uv`](https://docs.astral.sh/uv/) script (installs numpy +
matplotlib on first run):

```bash
uv run harness/plot_metrics.py metrics.txt -o metrics.png
./cpp/museproc data/session1.jsonl | uv run harness/plot_metrics.py -
```

(Or `pip install -r harness/requirements.txt` and run with plain `python3`.)

## How it works

`museproc` streams the JSONL line by line and reconstructs the EEG onto a regular
256 Hz grid:

1. **Ingest** (`cpp/jsonl_reader.*`): parse `eeg` records, unwrap the 16-bit
   packet `index`, group the four electrodes of each packet into a 12×4 block,
   and place it on the grid (missing packets leave NaN gaps). The `timestamp`
   field is ignored — it is a float32-quantized device clock; the integer packet
   counter gives exact spacing.
2. **Filter**: an optional causal 60 Hz biquad notch (Direct-Form-II transposed),
   reset across dropouts. This is the streaming-appropriate replacement for the
   offline zero-phase filter used in the Python reference.
3. **Metrics** (`cpp/metrics.*`): every `--hop` samples, compute per-channel
   signal quality (RMS thresholds good&lt;50 / marginal&lt;100 / poor µV), quality
   weights (drop up to the 2 worst poor channels; good=1, marginal=0.5), Morlet
   band powers, and the four output metrics.

The algorithms are a streaming port of the offline reference in `reference/`
(originally `analysis/utils.py` / `analysis/eeg.py` from the Nouscope project).
`harness/golden.py` is a numpy-only re-implementation used to validate the port;
its metrics match `museproc --no-notch` to 5–6 significant figures.

## Data

Recorded sessions live in `data/` (gitignored — biometric data). `session1–3`
are real Muse-340C captures; `cutofftest` and `simulatedtestsession` exercise
backlog-trim and synthetic edge cases.

## Provenance

This repository began as **Nouscope**, a browser (Vite/WebGL) Muse visualizer.
It was stripped down to this C++ processor; the original JS app and full Python
analysis suite remain in git history.
