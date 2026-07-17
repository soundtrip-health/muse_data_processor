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
cd cpp && make                       # -> cpp/museproc (g++ -O2 -std=c++17)
./cpp/museproc data/session1.jsonl   # stream metrics to stdout
uv run harness/plot_metrics.py metrics.txt   # 4-panel plot
```

No CMake, no third-party C++ libraries. Python harness runs via `uv` (inline
deps) or `pip install -r harness/requirements.txt`.

## Layout

| Path | Role |
|---|---|
| `cpp/jsonl_reader.*` | JSONL parse, 16-bit counter unwrap, packet→256 Hz grid, causal notch, rolling buffers |
| `cpp/metrics.*` | Morlet band power, signal quality, quality weights, sample entropy / MSE, symmetry |
| `cpp/main.cpp` | CLI, emit loop, output formatting |
| `harness/plot_metrics.py` | plot the metrics stream |
| `harness/golden.py` | numpy-only reference; validates the C++ port (matches `--no-notch` to ~6 sig figs) |
| `reference/utils.py`, `reference/eeg.py` | original offline algorithm source-of-truth (NOT runnable standalone) |
| `data/` | recorded sessions (gitignored, biometric) |

## Working rules

- **`reference/utils.py` + `reference/eeg.py` are the algorithm source-of-truth.**
  When changing signal processing, keep the C++ faithful to them (or update both
  intentionally) and re-check parity with `harness/golden.py`.
- Muse electrode order is fixed: `0=TP9, 1=AF7, 2=AF8, 3=TP10`. All four scalp
  channels are processed; there is no separate reference channel.
- The processor is **streaming and causal** — no look-ahead, no zero-phase
  filtering. Preserve that for the eventual live BLE source.
- `graphify-out/` indexes the *old* JS codebase and is now stale. Run
  `graphify update .` to regenerate against the C++ tree if you rely on it.
