# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Plot a museproc metrics stream.

Reads the text time-series emitted by the C++ processor (space- or comma-
separated, with a leading `# t entropy theta_alpha alpha_symmetry quality`
header) and draws a 4-panel figure. `nan` values render as gaps.

Usage:
    uv run harness/plot_metrics.py metrics.txt [-o out.png] [--show]
    ./cpp/museproc data/session1.jsonl | uv run harness/plot_metrics.py -

Without -o, the figure is saved into the data/ folder as data/<name>.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

COLS = ["t", "entropy", "theta_alpha", "alpha_symmetry", "quality"]
PANELS = [
    ("entropy", "MSE complexity", "#5B8FF9"),
    ("theta_alpha", "theta / alpha", "#61DDAA"),
    ("alpha_symmetry", "alpha symmetry\nln(AF8) - ln(AF7)", "#F6BD16"),
    ("quality", "data quality (0-1)", "#E8684A"),
]


def load(path):
    f = sys.stdin if path == "-" else open(path)
    rows = []
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < len(COLS):
            continue
        rows.append([float("nan") if p == "nan" else float(p) for p in parts[: len(COLS)]])
    if f is not sys.stdin:
        f.close()
    return np.asarray(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="metrics file, or - for stdin")
    ap.add_argument("-o", "--out", help="output PNG path (default <input>.png)")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    data = load(args.input)
    if data.size == 0:
        sys.exit("no data rows found")
    t = data[:, 0]

    if not args.show:
        matplotlib.use("Agg")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 8), sharex=True)
    for ax, (key, label, color) in zip(axes, PANELS):
        y = data[:, COLS.index(key)]
        ax.plot(t, y, color=color, lw=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"museproc metrics — {args.input}", fontsize=11)
    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        if args.out:
            out = args.out
        else:
            # Default: save into the data/ folder (gitignored scratch area).
            stem = Path(args.input).stem if args.input != "-" else "metrics"
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            out = str(data_dir / f"{stem}.png")
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
