# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`muse_data_processor` is a streaming **C++ command-line tool** (`museproc`) that
reads a raw Muse EEG stream — a recorded `.jsonl` session now, live BLE later —
and emits a real-time text time-series of four biomarkers:
`t · entropy · theta_alpha · alpha_symmetry · quality`. See `README.md` for the
metric definitions and CLI.

It was converted from **Nouscope**, a browser EEG visualizer; the old JS app and
Python analysis suite are gone from the tree but remain in git history.

## Build & run

```bash
cd cpp && make                                     # -> cpp/museproc (g++ -O2 -std=c++17)
./cpp/museproc data/session1.jsonl > data/session1.metrics.txt
uv run harness/plot_metrics.py data/session1.metrics.txt   # -> data/session1.metrics.png
```

No CMake, no third-party C++ libraries. Python harness runs via `uv` (inline
deps) or `pip install -r harness/requirements.txt`.

## Layout

| Path | Role |
|---|---|
| `cpp/jsonl_reader.*` | JSONL parse, 16-bit counter unwrap, packet→256 Hz grid, causal notch, rolling buffers |
| `cpp/metrics.*` | Morlet band power, signal quality, quality weights, sample entropy / MSE, symmetry |
| `cpp/main.cpp` | CLI, emit loop, output formatting |
| `harness/plot_metrics.py` | plot the metrics stream (saves into `data/` by default) |
| `harness/golden.py` | numpy-only reference implementation; validates the C++ port (matches `--no-notch` to ~6 sig figs) and is the algorithm source-of-truth |
| `data/` | recorded sessions + generated metrics/plots (gitignored except `.gitkeep`) |

## Working rules

- **`harness/golden.py` is the algorithm source-of-truth.** When changing signal
  processing, keep the C++ and `golden.py` in agreement and re-check parity. The
  original offline Nouscope analysis (`analysis/utils.py` / `eeg.py`) lives in
  git history if you need the fuller reference.
- Muse electrode order is fixed: `0=TP9, 1=AF7, 2=AF8, 3=TP10`. All four scalp
  channels are processed; there is no separate reference channel.
- The processor is **streaming and causal** — no look-ahead, no zero-phase
  filtering. Preserve that for the eventual live BLE source.
- `graphify-out/` indexes the *old* JS codebase and is now stale. Run
  `graphify update .` to regenerate against the C++ tree if you rely on it.
