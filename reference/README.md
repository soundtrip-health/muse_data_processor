# reference/

The original **offline** algorithm implementations from the Nouscope project,
kept as the source-of-truth for the C++ port in `../cpp/`.

- `utils.py` — full offline pipeline (JSONL load, packet→grid reconstruction,
  signal quality, band power, multiscale entropy). The C++ streaming processor
  mirrors these functions.
- `eeg.py` — EEG ingestion + multiscale-entropy helpers.

These are **not runnable standalone** here: they import scipy / antropy / pandas
and a `.config` package that was removed in the strip. They exist for reading and
provenance. For a runnable, validated reference use `../harness/golden.py`
(numpy-only), which reproduces the C++ metrics to ~6 significant figures.
