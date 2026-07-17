"""EEG ingestion and multiscale entropy.

Loads Muse-style EEG from a JSONL sensor stream, drops noisy channels by RMS,
and computes multiscale sample entropy (MSE) — coarse-grain by non-overlapping
block averages at scales 1..N, then sample entropy at each scale.

Public surface used by `chaos.eeg_render`:
  - load_eeg_from_jsonl(path, line_freq=60) -> eeg_df (float32, cols = electrodes)
  - slice_window(eeg_df, t_center, window_sec) -> eeg_df slice
  - clean_channels(window_df) -> (clean_df, dropped, kept_over_threshold)
  - multiscale_entropy(signal, scales) -> np.ndarray[len(scales)]
  - entropy_to_complexity(sampen) -> float in [0, 1]

Tunables (`rms_threshold_uv`, `max_drop_channels`, entropy mapping range)
default to values loaded from `chaos/config.yaml`; pass explicit kwargs to
override per-call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

import antropy

from .config import load_config

EEG_FS = 256
EEG_DT = 1 / EEG_FS
_MUSE_ELECTRODES = {0: "TP9", 1: "AF7", 2: "AF8", 3: "TP10"}

_ZIP_MAGIC = b"PK\x03\x04"


def _iter_jsonl_lines(path: str | Path):
    """Yield decoded lines from a `.jsonl` file or from the first `.jsonl`
    entry (or first entry) of a zip archive."""
    import io
    import zipfile

    with open(path, "rb") as fp:
        head = fp.read(4)
        if head == _ZIP_MAGIC:
            fp.seek(0)
            with zipfile.ZipFile(fp) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    raise ValueError(f"zip archive is empty: {path}")
                jsonl = [n for n in names if n.endswith(".jsonl")]
                entry = jsonl[0] if jsonl else names[0]
                with zf.open(entry) as zfp:
                    for line in io.TextIOWrapper(zfp, encoding="utf-8"):
                        yield line
            return
        fp.seek(0)
        for line in io.TextIOWrapper(fp, encoding="utf-8"):
            yield line


def load_eeg_from_jsonl(path: str | Path, line_freq: int | None = 60) -> pd.DataFrame:
    """Load EEG records from a JSONL sensor file.

    Accepts either a `.jsonl` file or a `.zip` archive; for a zip, the first
    `.jsonl` entry (or first entry if none match) is used. Ignores non-EEG
    records (gyro/accel/ppg). Interpolates NaN gaps from dropped packets and
    applies a notch filter at `line_freq` (set to None to skip). Returns a
    DataFrame indexed by relative time (s), with one column per electrode
    (TP9, AF7, AF8, TP10).
    """
    # Stream into per-electrode flat lists. Avoid per-packet DataFrames —
    # 220k 12-row DataFrames + concat peaks at 2 GB on a 40-min recording
    # while the final dense form is only a few MB.
    indices_by_e: dict[int, list[int]] = {e: [] for e in _MUSE_ELECTRODES}
    samples_by_e: dict[int, list[np.ndarray]] = {e: [] for e in _MUSE_ELECTRODES}

    for line in _iter_jsonl_lines(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") != "eeg":
            continue
        e = r["electrode"]
        if e not in indices_by_e:
            continue
        indices_by_e[e].append(r["index"])
        samples_by_e[e].append(np.asarray(r["samples"], dtype=np.float32))

    if not any(samples_by_e.values()):
        raise ValueError(f"no EEG records found on Muse electrodes (0-3) in {path}")

    seq_start = min(min(v) for v in indices_by_e.values() if v)

    series_by_name: dict[str, pd.Series] = {}
    for e in sorted(_MUSE_ELECTRODES):
        chunks = samples_by_e[e]
        if not chunks:
            continue
        idxs = np.asarray(indices_by_e[e], dtype=np.int64)
        order = np.argsort(idxs, kind="stable")
        idxs = idxs[order]
        chunks = [chunks[i] for i in order]

        lens = np.fromiter((c.shape[0] for c in chunks), dtype=np.int64, count=len(chunks))
        values = np.concatenate(chunks)
        # Reltime per sample = packet base (index step × 12 samples × EEG_DT)
        # plus per-sample offset within the packet.
        bases = (idxs - seq_start).astype(np.float64) * (EEG_DT * 12)
        reltimes = np.repeat(bases, lens)
        # Per-packet 0..len-1 offsets, concatenated, scaled by EEG_DT.
        within = np.concatenate([np.arange(int(L), dtype=np.float64) for L in lens])
        reltimes += within * EEG_DT

        s = pd.Series(values, index=pd.Index(reltimes, name="reltime"))
        if not s.index.is_unique:
            s = s.groupby(level=0).mean()
        elif not s.index.is_monotonic_increasing:
            s = s.sort_index()
        series_by_name[_MUSE_ELECTRODES[e]] = s

    eeg_df = pd.DataFrame(series_by_name)

    if eeg_df.isna().any().any():
        eeg_df = eeg_df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    if line_freq:
        b, a = sp_signal.iirnotch(line_freq, Q=30, fs=EEG_FS)
        for col in eeg_df.columns:
            eeg_df[col] = sp_signal.filtfilt(b, a, eeg_df[col].values)

    return eeg_df.astype(np.float32)


def slice_window(eeg_df: pd.DataFrame, t_center: float, window_sec: float) -> pd.DataFrame:
    """Return the slice of `eeg_df` covering [t_center - W/2, t_center + W/2]."""
    half = window_sec / 2.0
    t_lo, t_hi = t_center - half, t_center + half
    t = eeg_df.index.values
    mask = (t >= t_lo) & (t <= t_hi)
    if not mask.any():
        raise ValueError(
            f"window {t_lo:.1f}-{t_hi:.1f}s is outside EEG span "
            f"[{t[0]:.1f}, {t[-1]:.1f}]"
        )
    return eeg_df.iloc[mask]


def clean_channels(
    window_df: pd.DataFrame,
    rms_threshold_uv: float | None = None,
    max_drop_channels: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Drop the worst noisy channels (highest demeaned RMS over threshold),
    capped at `max_drop_channels` so we always retain enough channels to
    average even on heavily noisy recordings.

    Returns `(kept_df, dropped, kept_over_threshold)`:
      - `dropped`: channel name → RMS for channels actually removed (the
        top-RMS exceeders, up to `max_drop_channels`).
      - `kept_over_threshold`: channel name → RMS for channels that exceed
        the threshold but were retained because the drop cap was hit. A
        non-empty dict here signals a data-quality issue — the kept signal
        is still being averaged with noisy channels.

    Defaults for both kwargs come from `config.yaml` (eeg section).
    """
    cfg = load_config().eeg
    if rms_threshold_uv is None:
        rms_threshold_uv = cfg.rms_threshold_uv
    if max_drop_channels is None:
        max_drop_channels = cfg.max_drop_channels

    rms_by_col: dict[str, float] = {}
    for col in window_df.columns:
        x = window_df[col].values.astype(np.float64)
        x = x - x.mean()
        rms_by_col[col] = float(np.sqrt(np.mean(x * x)))

    over = sorted(
        ((c, r) for c, r in rms_by_col.items() if r > rms_threshold_uv),
        key=lambda cr: cr[1],
        reverse=True,
    )
    drop_set = {c for c, _ in over[:max_drop_channels]}
    dropped = {c: rms_by_col[c] for c in drop_set}
    kept_over_threshold = {c: r for c, r in over[max_drop_channels:]}
    kept_cols = [c for c in window_df.columns if c not in drop_set]
    return window_df[kept_cols], dropped, kept_over_threshold


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Non-overlapping block-average coarse graining at the given scale."""
    if scale <= 1:
        return x
    n = (len(x) // scale) * scale
    return x[:n].reshape(-1, scale).mean(axis=1)


def multiscale_entropy(x: np.ndarray, scales: Sequence[int]) -> np.ndarray:
    """Sample entropy at each of the given coarse-grain scales.

    `scales` is an explicit list like `[1, 3, 5, 7]` — the result is aligned
    with it. Uses antropy.sample_entropy (order=2, Chebyshev distance,
    default r).
    """
    out = np.empty(len(scales), dtype=np.float64)
    for i, s in enumerate(scales):
        out[i] = sample_entropy_at_scale(x, int(s))
    return out


def sample_entropy_at_scale(x: np.ndarray, scale: int) -> float:
    """Sample entropy after coarse-graining `x` at a single scale (≥1)."""
    if scale < 1:
        raise ValueError(f"scale must be ≥ 1, got {scale}")
    return float(antropy.sample_entropy(_coarse_grain(x, scale)))


def entropy_to_complexity(
    sampen: float, lo: float | None = None, hi: float | None = None
) -> float:
    """Linear map `sampen` in [lo, hi] → [0, 1], clipped.

    Defaults for `lo`/`hi` come from `config.yaml` (eeg section).
    """
    if lo is None or hi is None:
        cfg = load_config().eeg
        if lo is None:
            lo = cfg.entropy_complexity_lo
        if hi is None:
            hi = cfg.entropy_complexity_hi
    if not np.isfinite(sampen):
        return 0.0
    return float(np.clip((sampen - lo) / (hi - lo), 0.0, 1.0))
