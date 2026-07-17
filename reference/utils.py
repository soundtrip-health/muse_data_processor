"""Offline analysis helpers for Nouscope JSONL recordings.

This module hosts the data loading, signal processing, and plotting pipeline
used by `nouscope_analysis.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, find_peaks
from scipy.spatial.distance import pdist

EEG_FS = 256
PPG_FS = 64
IMU_FS = 52

# Band-power pipeline (mirrors EEGManager.js)
EEG_WIN = 256  # samples (1 s) — analysis window
EEG_HOP = 128  # samples (0.5 s) — 50% overlap, ~2 Hz
WAVELET_BANDS = {"theta": 6, "alpha": 10, "beta": 20, "gamma": 40}  # Hz
WAVELET_TAU = {"theta": 4, "alpha": 6, "beta": 6, "gamma": 6}  # cycles
DELTA_BINS = (1, 2, 3)  # Hz — sparse Hann-DFT for delta
AP_FIT_FREQS = (6, 10, 20, 40)
AP_REFIT_EVERY = 10  # windows (~5 s)
AP_SMOOTH = 0.30
AP_MIN_REFITS = 3  # warmup before output is non-zero
BAND_SMOOTH = 0.35  # output EMA
SQ_LOW, SQ_HIGH = 50.0, 100.0  # µV — good <50, marginal 50–100, poor >100

# Multiscale entropy (mirrors ComplexityManager.js)
MSE_SCALES = (1, 3, 5, 7, 9)
MSE_M = 2
MSE_R_COEF = 0.15
MSE_WIN = 2048  # 8 s @ 256 Hz
MSE_HOP_S = 5.0  # update interval

# Tempogram (mirrors EntrainmentManager.js, EEG side only)
TEMPO_WIN_S = 8.0
TEMPO_LO_HZ = 0.5
TEMPO_HI_HZ = 5.0
TEMPO_STEP_HZ = 0.1
TEMPO_DETREND_S = 0.5  # subtract centered moving average
TEMPO_HOP_S = 0.5  # ~2 Hz update

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
Q_WEIGHT = {"good": 1.0, "marginal": 0.5, "poor": 0.0}
PPG_INFRARED = 1  # ppgChannel 1 — mirrors EEGManager's choice for heart-rate


@dataclass
class Recording:
    """Parsed Nouscope JSONL recording. All time values are in seconds."""

    meta: dict
    eeg_packets: list[tuple[float, np.ndarray]]  # (t_s, (k, 4))
    ppg_packets: list[tuple[float, np.ndarray]]  # (t_s, (k,))
    imu_accel: list[tuple[float, np.ndarray]]  # (t_s, (3,))
    imu_gyro: list[tuple[float, np.ndarray]]  # (t_s, (3,))
    bands: pd.DataFrame  # t, delta..gamma
    hr: pd.DataFrame  # t, bpm
    mse: pd.DataFrame  # t, complexity, curve(list)
    entrain: pd.DataFrame  # t, idx
    duration_s: float


_COUNTER_MODULUS = 1 << 16  # muse-js index/sequenceId counters are 16-bit


def _unwrap_counter(state: dict, key, raw_value: int) -> int:
    """Correct a 16-bit hardware counter (`index`/`sequenceId`) for
    wraparound. `state` accumulates `{key: (prev_raw, wraps)}` across calls —
    call once per reading, in file order, per independent counter stream
    (e.g. one per electrode/ppgChannel). A backward jump of more than half
    the modulus is treated as one wrap, not a real time-reversal.

    We reconstruct time from this counter rather than the raw `timestamp`
    field: `timestamp` turned out to be a genuine device clock, but at the
    ~1.8e12 ms magnitude these recordings carry, it has clearly been through
    a float32 cast somewhere upstream — float32 only resolves ~131k-unit
    steps at that magnitude, so per-packet timestamp deltas (true spacing
    46.875 ms) mostly round to 0 and then jump in ~131k-unit multiples of
    that same 46.875 ms once enough real time accumulates. The counter is an
    exact integer with no such precision loss.
    """
    prev = state.get(key)
    wraps = 0 if prev is None else prev[1] + (1 if prev[0] - raw_value > _COUNTER_MODULUS // 2 else 0)
    state[key] = (raw_value, wraps)
    return raw_value + wraps * _COUNTER_MODULUS


def _stream_start_offset(t_ms) -> float:
    """Seconds to add to a stream's counter-relative time so it lands on the
    same absolute clock as the `t`-bearing derived streams (bands/hr/mse/
    entrain/music), instead of at a local `t=0`. `t_ms` is the first packet's
    own `t` field (ms since capture epoch) if the recording has one — added
    to raw records so a stream still anchors correctly even when the
    recorder's pre-record backlog (`BACKLOG_MAX_BYTES` in RecordingManager.js)
    has trimmed away the front of the session, and this packet isn't really
    the start of the capture. Recordings made before raw records carried `t`
    fall back to 0 (first sample = local zero, same as before) — fine for an
    untrimmed file, where that coincides with the true session start anyway.
    """
    return t_ms / 1000.0 if t_ms is not None else 0.0


def _reconstruct_eeg_packets(
    eeg_by_idx: dict[int, dict[int, np.ndarray]], eeg_t_by_idx: dict[int, float | None]
) -> list[tuple[float, np.ndarray]]:
    """Merge per-electrode packets sharing an (unwrapped) muse-js `index`
    into one (12, 4) block per packet. `index` advances once per packet-group
    (~21.33/s), so time is
    `start_offset + (index - first_index) * samples_per_packet / EEG_FS`,
    where `start_offset` anchors the first packet to the real capture clock
    (see `_stream_start_offset`). Groups missing an electrode (a dropped BLE
    notification) are skipped — they surface as a gap downstream, same as any
    other dropout."""
    idxs = sorted(eeg_by_idx)
    if not idxs:
        return []
    seq_start = idxs[0]
    start_offset = _stream_start_offset(eeg_t_by_idx.get(seq_start))
    packets = []
    for idx in idxs:
        chans = eeg_by_idx[idx]
        if len(chans) < 4:
            continue
        block = np.stack([chans[e] for e in range(4)], axis=1)  # (12, 4)
        t = start_offset + (idx - seq_start) * (block.shape[0] / EEG_FS)
        packets.append((t, block))
    return packets


def _reconstruct_indexed_packets(
    by_idx: dict[int, np.ndarray],
    samples_per_packet: int,
    fs: float,
    t_by_idx: dict[int, float | None] | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Packets from a single (unwrapped) index-keyed stream (e.g. one PPG
    channel), same reconstruction — and same absolute anchoring — as
    `_reconstruct_eeg_packets`."""
    idxs = sorted(by_idx)
    if not idxs:
        return []
    seq_start = idxs[0]
    start_offset = _stream_start_offset((t_by_idx or {}).get(seq_start))
    return [(start_offset + (idx - seq_start) * (samples_per_packet / fs), by_idx[idx]) for idx in idxs]


def _reconstruct_seq_packets(
    raw: list[tuple[int, float | None, np.ndarray]], samples_per_packet: int, fs: float
) -> list[tuple[float, np.ndarray]]:
    """Packets from an (unwrapped) `sequenceId`-keyed stream (accel/gyro),
    same absolute anchoring as `_reconstruct_eeg_packets`. `raw` entries are
    `(sequence_id, t_ms_or_None, block)`."""
    if not raw:
        return []
    raw = sorted(raw, key=lambda p: p[0])
    seq_start, t0_ms, _ = raw[0]
    start_offset = _stream_start_offset(t0_ms)
    return [(start_offset + (seq_id - seq_start) * (samples_per_packet / fs), block) for seq_id, _t, block in raw]


def load_jsonl(path: str | Path) -> Recording:
    """Parse a Nouscope recording. Raw sensor records (`eeg`/`ppg`/`accel`/
    `gyro`) mirror the native muse-js reading shape — a 16-bit
    `index`/`sequenceId` counter (unwrapped via `_unwrap_counter`, see its
    docstring for why we don't use the `timestamp` field despite it looking
    like the more natural clock) gives each stream's *relative* spacing. They
    also carry `t` (ms since capture epoch, the same clock the `t`-bearing
    derived streams — `bands`/`hr`/`mse`/`entrain` — use), which anchors each
    stream's first packet to the real capture clock instead of a local `t=0`
    (see `_stream_start_offset`). This matters whenever the recorder's
    pre-record backlog (`BACKLOG_MAX_BYTES` in RecordingManager.js) trims the
    front of a long session: the saved file's raw data can start well after
    true `t=0`, and without this anchor it would silently look like a fresh
    session starting at 0 — desynced from the correctly-timed derived series.
    Recordings made before raw records carried `t` fall back to first-sample
    = local zero, same as before; there can be a small offset between
    "recorded" and "recomputed" series in the overview plot for those.
    """
    path = Path(path)
    meta: dict = {}
    eeg_by_idx: dict[int, dict[int, np.ndarray]] = {}
    eeg_idx_state: dict[int, tuple[int, int]] = {}
    eeg_t_by_idx: dict[int, float | None] = {}
    ppg_ir_by_idx: dict[int, np.ndarray] = {}
    ppg_idx_state: dict[int, tuple[int, int]] = {}
    ppg_t_by_idx: dict[int, float | None] = {}
    accel_raw: list[tuple[int, float | None, np.ndarray]] = []
    accel_idx_state: dict[int, tuple[int, int]] = {}
    gyro_raw: list[tuple[int, float | None, np.ndarray]] = []
    gyro_idx_state: dict[int, tuple[int, int]] = {}
    bands_rows, hr_rows, mse_rows, entrain_rows = [], [], [], []

    with path.open() as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            tp = r.get("type")
            if tp == "meta":
                meta = r
            elif tp == "eeg":
                e = r["electrode"]
                idx = _unwrap_counter(eeg_idx_state, e, r["index"])
                eeg_by_idx.setdefault(idx, {})[e] = np.asarray(r["samples"], dtype=np.float64)
                eeg_t_by_idx.setdefault(idx, r.get("t"))
            elif tp == "ppg":
                ch = r["ppgChannel"]
                idx = _unwrap_counter(ppg_idx_state, ch, r["index"])
                if ch == PPG_INFRARED:
                    ppg_ir_by_idx[idx] = np.asarray(r["samples"], dtype=np.float64)
                    ppg_t_by_idx.setdefault(idx, r.get("t"))
            elif tp == "accel":
                idx = _unwrap_counter(accel_idx_state, 0, r["sequenceId"])
                accel_raw.append((idx, r.get("t"), np.array([[s["x"], s["y"], s["z"]] for s in r["samples"]])))
            elif tp == "gyro":
                idx = _unwrap_counter(gyro_idx_state, 0, r["sequenceId"])
                gyro_raw.append((idx, r.get("t"), np.array([[s["x"], s["y"], s["z"]] for s in r["samples"]])))
            elif tp == "bands":
                t = r["t"] / 1000.0
                bands_rows.append({"t": t, **{k: r[k] for k in ("delta", "theta", "alpha", "beta", "gamma")}})
            elif tp == "hr":
                hr_rows.append({"t": r["t"] / 1000.0, "bpm": r["bpm"]})
            elif tp == "mse":
                mse_rows.append({"t": r["t"] / 1000.0, "complexity": r["complexity"], "curve": r["curve"]})
            elif tp == "entrain":
                entrain_rows.append({"t": r["t"] / 1000.0, "idx": r["idx"]})

    eeg_packets = _reconstruct_eeg_packets(eeg_by_idx, eeg_t_by_idx)
    ppg_packets = _reconstruct_indexed_packets(ppg_ir_by_idx, samples_per_packet=6, fs=PPG_FS, t_by_idx=ppg_t_by_idx)
    imu_accel = _reconstruct_seq_packets(accel_raw, samples_per_packet=3, fs=IMU_FS)
    imu_gyro = _reconstruct_seq_packets(gyro_raw, samples_per_packet=3, fs=IMU_FS)

    last_t = max((p[0] for p in eeg_packets), default=0.0)
    if eeg_packets:
        last_t = max(last_t, eeg_packets[-1][0] + len(eeg_packets[-1][1]) / EEG_FS)

    return Recording(
        meta=meta,
        eeg_packets=eeg_packets,
        ppg_packets=ppg_packets,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        bands=pd.DataFrame(bands_rows),
        hr=pd.DataFrame(hr_rows),
        mse=pd.DataFrame(mse_rows),
        entrain=pd.DataFrame(entrain_rows),
        duration_s=last_t,
    )


def packets_to_grid(
    packets: list[tuple[float, np.ndarray]],
    fs: float,
    duration_s: float,
    n_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place each packet's samples on a regular `fs`-Hz grid using its
    timestamp as the index of its first sample. Overlapping samples (clock
    jitter, duplicates) are kept from whichever packet writes first; gaps
    remain NaN.

    Returns (times_s, data) where data has shape (N,) for 1-D streams or
    (N, n_channels) for multi-channel ones.
    """
    if not packets:
        return np.empty(0), np.empty(0)
    if n_channels is None:
        sample = packets[0][1]
        n_channels = sample.shape[1] if sample.ndim == 2 else 1

    n_total = int(np.ceil(duration_s * fs)) + 12
    shape = (n_total,) if n_channels == 1 else (n_total, n_channels)
    data = np.full(shape, np.nan, dtype=np.float64)

    for t, samples in packets:
        idx = int(round(t * fs))
        k = samples.shape[0]
        end = idx + k
        if end > n_total:
            extra = end - n_total + 64
            pad_shape = (extra,) if n_channels == 1 else (extra, n_channels)
            data = np.concatenate([data, np.full(pad_shape, np.nan)])
            n_total += extra
        # Only write into NaN slots so an earlier packet wins on overlap
        slot = data[idx:end]
        mask = np.isnan(slot) if slot.ndim == 1 else np.isnan(slot).any(axis=-1)
        if slot.ndim == 1:
            slot[mask] = samples[mask]
        else:
            slot[mask] = samples[mask]

    times = np.arange(data.shape[0]) / fs
    return times, data


def gap_summary(data: np.ndarray, fs: float) -> dict:
    """Report missing-sample statistics for diagnostics."""
    flat = data if data.ndim == 1 else np.isnan(data).any(axis=-1)
    bad = flat if data.ndim == 1 and data.dtype == bool else np.isnan(flat) if data.ndim == 1 else flat
    if data.ndim == 1:
        bad = np.isnan(data)
    n_bad = int(bad.sum())
    # Run-length encode the gaps to find the longest dropout
    if n_bad == 0:
        max_gap = 0
    else:
        diff = np.diff(bad.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if bad[0]:
            starts = np.concatenate([[0], starts])
        if bad[-1]:
            ends = np.concatenate([ends, [len(bad)]])
        max_gap = int((ends - starts).max()) if len(starts) else 0
    return {
        "samples": int(data.shape[0]),
        "missing": n_bad,
        "missing_pct": 100.0 * n_bad / max(1, data.shape[0]),
        "max_gap_samples": max_gap,
        "max_gap_s": max_gap / fs,
    }


def interpolate_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate runs of NaNs ≤ `max_gap` samples; leave longer
    gaps as NaN so downstream windows skip them."""
    x = x.copy()
    n = len(x)
    isnan = np.isnan(x)
    if not isnan.any():
        return x
    diff = np.diff(isnan.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if isnan[0]:
        starts = np.concatenate([[0], starts])
    if isnan[-1]:
        ends = np.concatenate([ends, [n]])
    for s, e in zip(starts, ends):
        if e - s > max_gap or s == 0 or e == n:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def signal_quality(eeg: np.ndarray, fs: float = EEG_FS, win_s: float = 1.0) -> pd.DataFrame:
    """Per-channel RMS-based quality classification, sampled every ~0.25 s.
    Mirrors EEGManager `_updateSignalQuality` (good <50 µV, marginal 50–100, poor >100)."""
    win = int(win_s * fs)
    hop = win // 4
    n = eeg.shape[0]
    rows = []
    for i in range(win, n + 1, hop):
        block = eeg[i - win : i]  # (win, 4)
        valid = ~np.isnan(block)
        rms = np.full(4, np.nan)
        for ch in range(4):
            v = block[valid[:, ch], ch]
            if len(v) >= 10:
                v = v - v.mean()
                rms[ch] = float(np.sqrt(np.mean(v * v)))
        labels = []
        for r in rms:
            if not np.isfinite(r):
                labels.append("poor")
            elif r < SQ_LOW:
                labels.append("good")
            elif r < SQ_HIGH:
                labels.append("marginal")
            else:
                labels.append("poor")
        rows.append(
            {
                "t": i / fs,
                **{f"rms_{c}": rms[k] for k, c in enumerate(CH_NAMES)},
                **{f"q_{c}": labels[k] for k, c in enumerate(CH_NAMES)},
            }
        )
    return pd.DataFrame(rows)


def _build_morlet_kernels(fs: float = EEG_FS):
    """Pre-build Morlet wavelet kernels for the four wavelet bands.
    Matches the BOSC convention used in EEGManager._precomputeKernels."""
    kernels = {}
    for band, f in WAVELET_BANDS.items():
        tau = WAVELET_TAU[band]
        sigma = tau / (2 * np.pi * f)
        amp = 1.0 / np.sqrt(sigma * np.sqrt(np.pi))
        half = int(np.ceil(3.0 * sigma * fs))
        n = 2 * half + 1
        t = (np.arange(n) - half) / fs
        gauss = amp * np.exp(-0.5 * (t / sigma) ** 2)
        kernels[band] = (
            gauss * np.cos(2 * np.pi * f * t),
            gauss * np.sin(2 * np.pi * f * t),
            half,
        )
    return kernels


def _build_delta_kernels(win: int = EEG_WIN):
    """Hann-weighted DFT twiddle factors at integer Hz bins for the delta band."""
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1))
    out = {}
    for k in DELTA_BINS:
        ang = 2 * np.pi * k * np.arange(win) / win
        out[k] = (hann * np.cos(ang), hann * np.sin(ang))
    return out


_MORLET = _build_morlet_kernels()
_DELTA = _build_delta_kernels()


def _channel_band_power(sig: np.ndarray) -> dict:
    """Raw band power for a single EEG_WIN-sample window. Returns nan for the
    whole dict if any sample is NaN — caller should skip these windows."""
    if np.isnan(sig).any():
        return {"delta": np.nan, "theta": np.nan, "alpha": np.nan, "beta": np.nan, "gamma": np.nan}
    delta = 0.0
    for k in DELTA_BINS:
        re, im = _DELTA[k]
        r = float(re @ sig)
        m = float(im @ sig)
        delta += r * r + m * m
    out = {"delta": delta}
    n = len(sig)
    for band, (re, im, half) in _MORLET.items():
        klen = len(re)
        # Convolve only at fully-contained centers (no edge artefacts)
        valid = n - 2 * half
        if valid <= 0:
            out[band] = 0.0
            continue
        # Construct sliding windows of length klen; vectorised dot products
        sw = np.lib.stride_tricks.sliding_window_view(sig, klen)
        rr = sw @ re
        ii = sw @ im
        out[band] = float(np.mean(rr * rr + ii * ii))
    return out


def _weights_at(sq_df: pd.DataFrame, sq_t: np.ndarray, t_centre: float) -> tuple[np.ndarray, float]:
    """Quality weights (and total weight) for the signal-quality window covering
    `t_centre`. Shared by the window-based consumers (`band_power_timeseries`,
    `mse_timeseries`, `eeg_tempogram_timeseries`): pick the latest quality labels
    at or before the window centre, then delegate to `_quality_weights`. Falls
    back to all-good when no quality frame exists."""
    if sq_t.size:
        i = max(0, int(np.searchsorted(sq_t, t_centre, side="right") - 1))
        labels = [sq_df.iloc[i][f"q_{c}"] for c in CH_NAMES]
    else:
        labels = ["good"] * 4
    return _quality_weights(labels)


def _quality_weights(labels: list[str]) -> tuple[np.ndarray, float]:
    """Drop up to 2 worst channels, weight the rest (good=1.0, marginal=0.5).
    Mirrors EEGManager._getChannelWeights with default badChannelThreshold='poor'."""
    score = {"good": 2, "marginal": 1, "poor": 0}
    cands = sorted([c for c in range(4) if score[labels[c]] <= 0], key=lambda c: score[labels[c]])
    drop = set(cands[:2])
    w = np.array([0.0 if c in drop else Q_WEIGHT[labels[c]] for c in range(4)])
    total = w.sum()
    if total > 0:
        return w / total, total
    keep = [c for c in range(4) if c not in drop]
    fb = np.zeros(4)
    if keep:
        fb[keep] = 1.0 / len(keep)
    return fb, 0.0


def band_power_timeseries(eeg: np.ndarray, sq_df: pd.DataFrame, fs: float = EEG_FS) -> pd.DataFrame:
    """Walk the EEG matrix in EEG_WIN windows with EEG_HOP hop, compute the
    full webapp pipeline (per-channel band power → quality-weighted average →
    aperiodic 1/f normalisation → EMA smoothing) and return a tidy DataFrame.
    Windows containing any NaN sample are skipped (timestamp is still emitted
    with NaN values so downstream consumers can detect dropouts)."""
    n = eeg.shape[0]
    band_names = ("delta", "theta", "alpha", "beta", "gamma")
    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 40.0}

    smoothed = {b: 0.0 for b in band_names}
    ap_a, ap_b = 0.0, -1.5
    ap_refits = 0
    win_count = 0

    # Precompute a fast lookup of quality labels keyed to window centre time
    sq_t = sq_df["t"].values if not sq_df.empty else np.array([])

    rows = []
    for end in range(EEG_WIN, n + 1, EEG_HOP):
        start = end - EEG_WIN
        t_centre = (start + EEG_WIN / 2) / fs

        # Per-channel band power
        ch_bands = []
        any_nan = False
        for ch in range(4):
            bp = _channel_band_power(eeg[start:end, ch])
            ch_bands.append(bp)
            if any(np.isnan(v) for v in bp.values()):
                any_nan = True

        if any_nan:
            rows.append({"t": t_centre, **{b: np.nan for b in band_names}, "valid": False})
            continue

        weights, total_w = _weights_at(sq_df, sq_t, t_centre)
        raw = {b: float(sum(ch_bands[c][b] * weights[c] for c in range(4))) if total_w > 0 else 0.0 for b in band_names}

        # Periodic aperiodic refit on quality-weighted Morlet powers at AP_FIT_FREQS
        if win_count % AP_REFIT_EVERY == 0 and total_w > 0:
            log_p = np.array([np.log10(max(raw[b], 1e-12)) for b in ("theta", "alpha", "beta", "gamma")])
            log_f = np.log10(np.array(AP_FIT_FREQS))
            slope, intercept = np.polyfit(log_f, log_p, 1)
            smooth = 1.0 if ap_refits == 0 else AP_SMOOTH
            ap_a = (1 - smooth) * ap_a + smooth * intercept
            ap_b = (1 - smooth) * ap_b + smooth * slope
            ap_refits += 1

        # 1/f-normalised relative power
        if ap_refits < AP_MIN_REFITS:
            norm = {b: 0.0 for b in band_names}
        else:
            ratios = {}
            for b in band_names:
                expected = 10 ** (ap_a + ap_b * np.log10(band_freqs[b]))
                ratios[b] = raw[b] / expected if raw[b] > 0 else 0.0
            tot = sum(ratios.values())
            norm = {b: (ratios[b] / tot if tot > 0 else 0.0) for b in band_names}

        for b in band_names:
            smoothed[b] += BAND_SMOOTH * (norm[b] - smoothed[b])

        rows.append({"t": t_centre, **{b: smoothed[b] for b in band_names}, "valid": True})
        win_count += 1

    return pd.DataFrame(rows)


def _sample_entropy(sig: np.ndarray, m: int, r: float) -> float:
    """SampEn (Richman & Moorman 2000) using vectorised pairwise Chebyshev
    distances. Templates indexed `[i:i+m]` for `i in [0, N-m)` so this matches
    the JS implementation in ComplexityManager._sampleEntropy exactly."""
    n = len(sig)
    if n < m + 2:
        return 0.0
    L = n - m
    Xm = np.lib.stride_tricks.sliding_window_view(sig, m)[:L]
    Xm1 = np.lib.stride_tricks.sliding_window_view(sig, m + 1)[:L]
    Dm = pdist(Xm, metric="chebyshev")
    Dm1 = pdist(Xm1, metric="chebyshev")
    B = int((Dm <= r).sum())
    A = int((Dm1 <= r).sum())
    if A == 0 or B == 0:
        return 0.0
    return float(-np.log(A / B))


def _coarse_grain(x: np.ndarray, tau: int) -> np.ndarray:
    if tau <= 1:
        return x
    n = (len(x) // tau) * tau
    return x[:n].reshape(-1, tau).mean(axis=1)


def mse_timeseries(
    eeg: np.ndarray,
    sq_df: pd.DataFrame,
    fs: float = EEG_FS,
    scales: tuple[int, ...] = MSE_SCALES,
    win: int = MSE_WIN,
    hop_s: float = MSE_HOP_S,
) -> pd.DataFrame:
    """Multiscale entropy over a sliding `win`-sample buffer, advanced every
    `hop_s` seconds. The signal is the quality-weighted 4-channel average; if
    any sample in the window is NaN we linearly interpolate up to 50 ms gaps
    (≤ 13 samples) and skip the window if longer dropouts remain."""
    n = eeg.shape[0]
    hop = int(hop_s * fs)
    sq_t = sq_df["t"].values if not sq_df.empty else np.array([])
    rows = []
    for end in range(win, n + 1, hop):
        start = end - win
        t_centre = (start + win / 2) / fs
        block = eeg[start:end]  # (win, 4)
        weights, total_w = _weights_at(sq_df, sq_t, t_centre)
        if total_w == 0:
            rows.append({"t": t_centre, "complexity": np.nan, **{f"s{s}": np.nan for s in scales}})
            continue
        sig = (block * weights).sum(axis=1)
        sig = interpolate_short_gaps(sig, max_gap=13)
        if np.isnan(sig).any():
            rows.append({"t": t_centre, "complexity": np.nan, **{f"s{s}": np.nan for s in scales}})
            continue
        sigma = float(sig.std())
        if sigma < 1e-6:
            rows.append({"t": t_centre, "complexity": 0.0, **{f"s{s}": 0.0 for s in scales}})
            continue
        r = MSE_R_COEF * sigma
        curve = np.array([_sample_entropy(_coarse_grain(sig, s), MSE_M, r) for s in scales])
        rows.append({"t": t_centre, "complexity": float(curve.mean()), **{f"s{s}": float(curve[k]) for k, s in enumerate(scales)}})
    return pd.DataFrame(rows)


def _build_tempo_kernels(n: int, fs: float):
    """Hann-weighted DFT kernels at TEMPO_LO_HZ … TEMPO_HI_HZ in TEMPO_STEP_HZ."""
    freqs = np.arange(TEMPO_LO_HZ, TEMPO_HI_HZ + TEMPO_STEP_HZ / 2, TEMPO_STEP_HZ)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    re = np.empty((len(freqs), n))
    im = np.empty((len(freqs), n))
    t = np.arange(n) / fs
    for k, f in enumerate(freqs):
        ang = 2 * np.pi * f * t
        re[k] = hann * np.cos(ang)
        im[k] = hann * np.sin(ang)
    return freqs, re, im


def _centred_moving_average_subtract(x: np.ndarray, win: int) -> np.ndarray:
    """Subtract a centred moving average of length `win` from `x`. Uses a
    cumulative-sum so it costs O(N), matching the prefix-sum form in
    EntrainmentManager._computeEEGNovelty."""
    n = len(x)
    half = win // 2
    pre = np.concatenate([[0.0], np.cumsum(x)])
    out = x.copy()
    lo = np.maximum(0, np.arange(n) - half)
    hi = np.minimum(n - 1, np.arange(n) + half)
    out -= (pre[hi + 1] - pre[lo]) / (hi - lo + 1)
    return out


def eeg_tempogram_timeseries(
    eeg: np.ndarray,
    sq_df: pd.DataFrame,
    fs: float = EEG_FS,
    win_s: float = TEMPO_WIN_S,
    hop_s: float = TEMPO_HOP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding tempogram of the quality-weighted EEG average. Returns
    (times, freqs, power[freq, time]). Windows with a NaN segment longer than
    50 ms remain NaN. The audio side is unavailable offline, so we only
    return the EEG tempogram — for an entrainment-style index you would need
    a synchronised audio recording."""
    n = eeg.shape[0]
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    freqs, re_k, im_k = _build_tempo_kernels(win, fs)
    detrend_win = int(TEMPO_DETREND_S * fs)
    sq_t = sq_df["t"].values if not sq_df.empty else np.array([])
    times, cols = [], []
    for end in range(win, n + 1, hop):
        start = end - win
        t_centre = (start + win / 2) / fs
        block = eeg[start:end]
        weights, total_w = _weights_at(sq_df, sq_t, t_centre)
        if total_w == 0:
            times.append(t_centre)
            cols.append(np.full(len(freqs), np.nan))
            continue
        sig = (block * weights).sum(axis=1)
        sig = interpolate_short_gaps(sig, max_gap=13)
        if np.isnan(sig).any():
            times.append(t_centre)
            cols.append(np.full(len(freqs), np.nan))
            continue
        sig = _centred_moving_average_subtract(sig, detrend_win)
        rr = re_k @ sig
        ii = im_k @ sig
        cols.append(rr * rr + ii * ii)
        times.append(t_centre)
    return np.array(times), freqs, np.array(cols).T  # (n_freqs, n_times)


def heart_rate_timeseries(
    ppg: np.ndarray,
    fs: float = PPG_FS,
    win_s: float = 6.0,
    hop_s: float = 1.0,
) -> pd.DataFrame:
    """Bandpass (0.5–3.5 Hz) the raw PPG, run scipy `find_peaks` on a 6 s
    rolling window every 1 s, take the median IBI as the BPM estimate.
    Mirrors the cadence of EEGManager's MSPTDfast detector but uses a
    lighter-weight peak finder — good enough for offline review."""
    n = len(ppg)
    if n == 0:
        return pd.DataFrame(columns=["t", "bpm"])
    sig = interpolate_short_gaps(ppg.copy(), max_gap=int(0.2 * fs))
    # IIR bandpass via filtfilt with a simple second-order Butterworth design
    from scipy.signal import butter

    bf, af = butter(2, [0.5, 3.5], btype="band", fs=fs)
    valid = ~np.isnan(sig)
    out = np.full_like(sig, np.nan)
    if valid.any():
        out[valid] = filtfilt(bf, af, sig[valid])
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    rows = []
    for end in range(win, n + 1, hop):
        start = end - win
        block = out[start:end]
        if np.isnan(block).any():
            continue
        block = block - block.mean()
        peaks, _ = find_peaks(block, distance=int(0.3 * fs), prominence=block.std() * 0.5)
        if len(peaks) < 3:
            continue
        ibi = np.diff(peaks) / fs
        bpm = 60.0 / np.median(ibi)
        if 30 <= bpm <= 200:
            rows.append({"t": (start + win / 2) / fs, "bpm": float(bpm)})
    return pd.DataFrame(rows)


def notch_eeg(eeg: np.ndarray, line_hz: float = 60.0, fs: float = EEG_FS) -> np.ndarray:
    """Apply a NaN-safe IIR notch (default mains 60 Hz). Each contiguous
    non-NaN segment is filtered independently so dropouts don't smear noise
    across the rest of the recording."""
    b, a = iirnotch(line_hz, Q=30, fs=fs)
    out = eeg.copy()
    n = eeg.shape[0]
    for ch in range(eeg.shape[1]):
        col = eeg[:, ch]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        # Walk runs of valid samples
        diff = np.diff(valid.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if valid[0]:
            starts = np.concatenate([[0], starts])
        if valid[-1]:
            ends = np.concatenate([ends, [n]])
        for s, e in zip(starts, ends):
            if e - s > len(b) * 4:
                out[s:e, ch] = filtfilt(b, a, col[s:e])
    return out


def analyse(path: str | Path) -> dict:
    """Run the full pipeline on one JSONL file and return all timeseries.

    Returns a dict with:
      meta, duration_s, eeg_times, eeg, ppg_times, ppg, gaps_eeg, gaps_ppg,
      sq, bands_computed, mse_computed, tempogram (times, freqs, power),
      hr_computed, and the recorded timeseries for comparison
      (bands_recorded, mse_recorded, hr_recorded, entrain_recorded).
    """
    rec = load_jsonl(path)
    eeg_t, eeg = packets_to_grid(rec.eeg_packets, EEG_FS, rec.duration_s, n_channels=4)
    ppg_t, ppg = packets_to_grid(rec.ppg_packets, PPG_FS, rec.duration_s, n_channels=1)

    eeg_clean = notch_eeg(eeg)
    sq = signal_quality(eeg_clean)
    bands_c = band_power_timeseries(eeg_clean, sq)
    mse_c = mse_timeseries(eeg_clean, sq)
    tt, tf, tp = eeg_tempogram_timeseries(eeg_clean, sq)
    hr_c = heart_rate_timeseries(ppg)

    return {
        "meta": rec.meta,
        "duration_s": rec.duration_s,
        "eeg_times": eeg_t,
        "eeg": eeg_clean,
        "ppg_times": ppg_t,
        "ppg": ppg,
        "gaps_eeg": gap_summary(eeg, EEG_FS),
        "gaps_ppg": gap_summary(ppg, PPG_FS),
        "sq": sq,
        "bands_computed": bands_c,
        "mse_computed": mse_c,
        "tempogram": {"times": tt, "freqs": tf, "power": tp},
        "hr_computed": hr_c,
        "bands_recorded": rec.bands,
        "mse_recorded": rec.mse,
        "hr_recorded": rec.hr,
        "entrain_recorded": rec.entrain,
    }


