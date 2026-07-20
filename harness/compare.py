"""Runs museproc and golden.py on the same session file and checks that they
agree. This is the actual regression test for museproc's signal processing:
golden.py is the source-of-truth reference (see its docstring), and museproc
must match it to within the precision museproc's own text output can
represent (~6 significant figures).

Usage:
  python3 harness/compare.py data/session.jsonl [--ppg] [--imu] [--hop N]
                                                  [--entropy-stride N] [--museproc PATH]

theta_alpha, alpha_symmetry, quality, and --ppg/--imu are cheap and checked
on EVERY row. entropy is an O(n^2) pure-Python sample-entropy recomputed
fresh per row (~5s/row) and is instead checked on a sampled subset —
--entropy-stride N takes every Nth row (default: spread ~20 samples across
the whole file). Pass --entropy-stride 1 to check entropy exhaustively too
(slow: rows * ~5s).

Exits 0 and prints a summary if every checked value matches; exits 1 and
prints the first mismatches otherwise.
"""
import sys
from pathlib import Path

import golden

HERE = Path(__file__).resolve().parent


def close(cpp_str, py_val):
    """True if museproc's printed value and golden.py's float agree to
    within museproc's own %.6g display precision (see cpp/main.cpp's fmt())."""
    if cpp_str == 'nan':
        return py_val != py_val  # NaN
    if py_val != py_val:
        return False
    c = float(cpp_str)
    tol = max(abs(c), abs(py_val)) * 1e-5 + 1e-6
    return abs(c - py_val) <= tol


def report(name, cpp_s, py_v):
    flag = '' if close(cpp_s, py_v) else '  <-- MISMATCH'
    print(f"    {name:>14s}  museproc={cpp_s:<14s} golden.py={py_v:<14.6g}{flag}")


def main():
    import subprocess
    args = sys.argv[1:]
    want_ppg = '--ppg' in args
    want_imu = '--imu' in args
    hop = 128
    if '--hop' in args:
        i = args.index('--hop'); hop = int(args[i + 1]); args = args[:i] + args[i + 2:]
    entropy_stride = None
    if '--entropy-stride' in args:
        i = args.index('--entropy-stride'); entropy_stride = int(args[i + 1]); args = args[:i] + args[i + 2:]
    museproc = HERE.parent / 'cpp' / 'museproc'
    if '--museproc' in args:
        i = args.index('--museproc'); museproc = Path(args[i + 1]); args = args[:i] + args[i + 2:]
    args = [a for a in args if a not in ('--ppg', '--imu')]
    if not args:
        print(__doc__)
        sys.exit(1)
    path = args[0]

    print("# this can take up to ~3 minutes", flush=True)

    cmd = [str(museproc), path, '--no-notch', '--hop', str(hop)]
    if want_ppg: cmd.append('--ppg')
    if want_imu: cmd.append('--imu')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"museproc failed: {proc.stderr}", file=sys.stderr)
        sys.exit(1)
    rows = [l.split() for l in proc.stdout.splitlines() if not l.startswith('#')]

    grid, ppg_events, accel_events, gyro_events, commit_points = golden.load_session(path)
    Ls = golden.emit_schedule(commit_points, hop=hop, win=256)

    if len(rows) != len(Ls):
        print(f"ROW COUNT MISMATCH: museproc emitted {len(rows)} rows, "
              f"golden.py's schedule expects {len(Ls)} — cannot align, aborting.")
        sys.exit(1)

    if entropy_stride is None:
        entropy_stride = max(1, len(rows) // 20)

    fast_cols = ['theta_alpha', 'alpha_symmetry', 'quality']
    if want_ppg: fast_cols.append('ppg')
    if want_imu: fast_cols += ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    mismatches = []

    # Fast pass: every row, no entropy.
    for r, L in zip(rows, Ls):
        _, ta, sym, q = golden.metrics_at(grid, L, want_entropy=False)
        py_vals = [ta, sym, q]
        if want_ppg:
            py_vals.append(golden.hold_at(ppg_events, L))
        if want_imu:
            ax, ay, az = golden.hold_at(accel_events, L, 3)
            gx, gy, gz = golden.hold_at(gyro_events, L, 3)
            py_vals += [ax, ay, az, gx, gy, gz]
        cpp_vals = [r[2], r[3], r[4]] + (r[5:] if (want_ppg or want_imu) else [])
        if not all(close(c, p) for c, p in zip(cpp_vals, py_vals)):
            mismatches.append((r[0], fast_cols, cpp_vals, py_vals))

    # Sampled pass: entropy only, every entropy_stride-th row.
    entropy_checked = 0
    for r, L in list(zip(rows, Ls))[::entropy_stride]:
        ent, *_ = golden.metrics_at(grid, L, want_entropy=True)
        entropy_checked += 1
        if not close(r[1], ent):
            mismatches.append((r[0], ['entropy'], [r[1]], [ent]))

    print(f"# museproc vs golden.py — {path}")
    print(f"#   {len(rows)} rows checked exhaustively for: {', '.join(fast_cols)}")
    print(f"#   {entropy_checked} rows (stride={entropy_stride}) checked for: entropy")

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} mismatching row(s)")
        for t, cols, cpp_vals, py_vals in mismatches[:10]:
            print(f"  t={t}")
            for name, cpp_s, py_v in zip(cols, cpp_vals, py_vals):
                report(name, cpp_s, py_v)
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        sys.exit(1)
    else:
        print("\nPASS: all checked values agree")


if __name__ == '__main__':
    main()
