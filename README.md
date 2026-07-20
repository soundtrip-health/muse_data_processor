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

### Optional PPG / IMU columns

`--ppg` and `--imu` append raw sensor columns to each row. Both sensors report
faster than the metrics output cadence (`--hop`), so these are a **hold**, not
a resample or average: each row carries whatever sample was most recently seen
at that point in the stream, and every other sample in between rows is
discarded. Values are `nan` until the first matching record has been seen.

| Flag | Adds | Source |
|---|---|---|
| `--ppg` | `ppg` | last sample of the most recent `ppgChannel:1` (infrared) packet |
| `--imu` | `accel_x accel_y accel_z gyro_x gyro_y gyro_z` | last `{x,y,z}` sample of the most recent `accel`/`gyro` packet |

## Input

The tool reads a **Muse JSONL recording**: one JSON object per line. `eeg`
records are always used; `ppg` and `accel`/`gyro` records are additionally
read when `--ppg`/`--imu` are passed. All other record types (`bands`, `hr`,
`mse`, `entrain`, `meta`) are ignored. A minimal valid file looks like:

```jsonl
{"type":"meta","electrodeNames":["TP9","AF7","AF8","TP10"],"sampleRates":{"eeg":256}}
{"type":"eeg","index":0,"electrode":0,"samples":[ ... 12 floats, µV ... ]}
{"type":"eeg","index":0,"electrode":1,"samples":[ ... 12 ... ]}
{"type":"eeg","index":0,"electrode":2,"samples":[ ... 12 ... ]}
{"type":"eeg","index":0,"electrode":3,"samples":[ ... 12 ... ]}
{"type":"eeg","index":1,"electrode":0,"samples":[ ... ]}
```

Each `eeg` record carries 12 samples for one of the four electrodes; the four
electrodes of a packet share the same 16-bit `index` (which wraps at 65536).
EEG is 256 Hz, 12 samples per packet. An optional `meta` first line may declare
`sampleRates.eeg`; otherwise 256 Hz is assumed (override with `--fs`). This is
the format produced by the Muse recorders in the `soundtrip` toolchain.

**No recordings ship with the repo** — `data/` is empty on a fresh clone (kept
by a `.gitkeep`, everything else in it is gitignored). Drop your own `.jsonl`
into `data/` and point the tool at it. The examples below use `data/session.jsonl`
as a stand-in for your file.

## Build

Requires a C++17 compiler (`g++`). No CMake, no external libraries.

```bash
cd cpp && make          # produces cpp/museproc
```

## Run

```bash
./cpp/museproc data/session.jsonl > data/session.metrics.txt   # batch
./cpp/museproc data/session.jsonl --realtime                   # pace to wall-clock
cat data/session.jsonl | ./cpp/museproc -                      # stream from stdin
```

Replace `data/session.jsonl` with your own recording. Save metrics files into
`data/` — that folder is the gitignored scratch area for both recordings and
generated output.

Options:

| Flag | Default | Effect |
|---|---|---|
| `--hop N` | 128 | samples between output rows (128 ≈ 2 Hz at 256 Hz) |
| `--csv` | off | comma-separated output instead of space |
| `--realtime` | off | sleep so rows emit at wall-clock time (emulates a live stream) |
| `--no-notch` | on | disable the causal 60 Hz mains notch |
| `--line-hz F` | 60 | mains frequency for the notch |
| `--fs F` | 256 | EEG sample rate |
| `--ppg` | off | append a raw PPG (infrared) sample column, held to the output cadence |
| `--imu` | off | append raw accel/gyro x/y/z columns, held to the output cadence |

## Plot (test harness)

`harness/plot_metrics.py` draws a 4-panel figure from the tool's output. It is a
self-contained [`uv`](https://docs.astral.sh/uv/) script (installs numpy +
matplotlib on first run):

```bash
uv run harness/plot_metrics.py data/session.metrics.txt   # -> data/session.metrics.png
./cpp/museproc data/session.jsonl | uv run harness/plot_metrics.py -
```

Without `-o`, the figure is written into `data/` as `data/<name>.png`. (Or
`pip install -r harness/requirements.txt` and run with plain `python3`.)

## Testing against the golden reference

`harness/golden.py` is a numpy-only re-implementation of the same algorithm
`museproc` runs, used to validate the C++ port (see "How it works" below).
Run standalone, it only computes metrics for timestamps you pass explicitly:

```bash
python3 harness/golden.py data/session.jsonl --ppg --imu 5 30 90   # spot-check t=5s, 30s, 90s
```

For an actual pass/fail test, use `harness/compare.py`: it runs `museproc`
and `golden.py` on the same file and diffs every row.

```bash
python3 harness/compare.py data/session.jsonl --ppg --imu
```

`theta_alpha`, `alpha_symmetry`, `quality`, and `--ppg`/`--imu` are cheap and
checked on every row. `entropy` is an O(n²) sample-entropy recomputed from
scratch in pure Python (~5s/row), so it's checked on a sampled subset instead
(`--entropy-stride N`, default spreads ~20 samples across the file — pass
`--entropy-stride 1` to check it exhaustively too, slowly).

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

The algorithms are a streaming port of the original Nouscope offline analysis
(`analysis/utils.py` / `eeg.py`, preserved in git history). `harness/golden.py`
is a self-contained numpy re-implementation used to validate the port — its
metrics match `museproc --no-notch` to 5–6 significant figures.
