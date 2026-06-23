#!/usr/bin/env python3
"""
Validate multi-node training runs against a golden benchmark.

Compares the last train_loss and val_loss, and wall-clock training duration
from each benchmark-025_*node subdir against reference values.
A run PASSES only if ALL checked metrics are within tolerance.

Usage:
    python validate_benchmark.py /path/to/user/benchmark/dir
    python validate_benchmark.py /path/to/user/benchmark/dir --golden golden_benchmark.json
    python validate_benchmark.py /path/to/user/benchmark/dir --rel-tol 0.02 --time-tol 0.15
    python validate_benchmark.py /path/to/user/benchmark/dir --report report.txt
    python validate_benchmark.py /path/to/user/benchmark/dir --no-time

Exit codes:
    0  — all runs passed
    1  — one or more runs failed
    2  — usage / IO error
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_GOLDEN   = Path(__file__).parent / "golden_benchmark.json"
DEFAULT_REL_TOL  = 0.01   # 1%
DEFAULT_TIME_TOL = 0.10   # 10%


def parse_node_count(dirname: str) -> int:
    m = re.search(r"_(\d+)node$", dirname)
    return int(m.group(1)) if m else -1


def load_run_data(log_dir: Path, tags: list[str]) -> tuple[dict, float | None]:
    """
    Returns:
      - scalars: tag -> {"step": int, "value": float} | None
      - duration_seconds: float | None  (max wall_time - min wall_time across all scalar events)
    """
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    scalars = {}
    for tag in tags:
        if tag not in available:
            scalars[tag] = None
        else:
            events = ea.Scalars(tag)
            last = events[-1]
            scalars[tag] = {"step": last.step, "value": last.value}

    all_wall_times: list[float] = []
    for tag in available:
        events = ea.Scalars(tag)
        if events:
            all_wall_times.append(events[0].wall_time)
            all_wall_times.append(events[-1].wall_time)

    duration = (max(all_wall_times) - min(all_wall_times)) if len(all_wall_times) >= 2 else None
    return scalars, duration


def check_tolerance(user_val: float, ref_val: float, rel_tol: float) -> tuple[bool, float, float]:
    """Returns (passed, abs_diff, rel_diff_pct)."""
    abs_diff = abs(user_val - ref_val)
    rel_diff = abs_diff / abs(ref_val) if ref_val != 0 else float("inf")
    return rel_diff <= rel_tol, abs_diff, rel_diff * 100


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def fmt_check(passed: bool, user_val: float, ref_val: float, abs_diff: float, rel_pct: float) -> str:
    status = "✓ PASS" if passed else "✗ FAIL"
    return (
        f"{status}  user={user_val:.6f}  ref={ref_val:.6f}  "
        f"Δ={abs_diff:.6f} ({rel_pct:.3f}%)"
    )


def fmt_time_check(passed: bool, user_s: float, ref_s: float, rel_pct: float) -> str:
    status = "✓ PASS" if passed else "✗ FAIL"
    return (
        f"{status}  user={fmt_duration(user_s)}  ref={fmt_duration(ref_s)}  "
        f"Δ={rel_pct:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate benchmark-025_*node TensorBoard runs against golden values."
    )
    parser.add_argument("dirpath",    type=Path,  help="Root directory with user's benchmark-025_*node subdirs")
    parser.add_argument("--golden",   type=Path,  default=DEFAULT_GOLDEN,   help="Path to golden_benchmark.json")
    parser.add_argument("--rel-tol",  type=float, default=DEFAULT_REL_TOL,  help="Relative tolerance for loss values (default: 0.01 = 1%%)")
    parser.add_argument("--time-tol", type=float, default=DEFAULT_TIME_TOL, help="Relative tolerance for duration (default: 0.10 = 10%%)")
    parser.add_argument("--no-time",  action="store_true",                  help="Skip duration validation entirely")
    parser.add_argument("--report",   type=Path,  default=None,             help="Optional path to write a plain-text report")
    args = parser.parse_args()

    if not args.golden.is_file():
        print(f"ERROR: golden file not found: {args.golden}", file=sys.stderr)
        sys.exit(2)

    with open(args.golden) as f:
        golden: dict = json.load(f)

    root = args.dirpath
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    subdirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"benchmark-025_\d+node$", d.name)],
        key=lambda d: parse_node_count(d.name),
    )

    if not subdirs:
        print("No benchmark-025_*node directories found.", file=sys.stderr)
        sys.exit(2)

    lines = []
    sep = "═" * 100

    def emit(line=""):
        lines.append(line)
        print(line)

    emit(sep)
    emit(f"  PARADIS Multi-Node Benchmark Validator")
    emit(f"  Loss tolerance  → relative: {args.rel_tol*100:.1f}%")
    if not args.no_time:
        emit(f"  Time tolerance  → relative: {args.time_tol*100:.1f}%")
    else:
        emit(f"  Duration validation: disabled")
    emit(f"  Golden reference: {args.golden}")
    emit(f"  User results:     {root}")
    emit(sep)

    tags = ["train_loss", "val_loss"]
    overall_pass = True
    n_pass = n_fail = n_skip = 0

    for subdir in subdirs:
        nodes = parse_node_count(subdir.name)
        ref_entry = golden.get(str(nodes))

        emit(f"\n  ┌─ {subdir.name}  ({nodes} node{'s' if nodes > 1 else ''})")

        if ref_entry is None:
            emit(f"  │  ⚠  No golden reference for {nodes} node(s) — skipping")
            n_skip += 1
            emit(f"  └{'─'*60}")
            continue

        try:
            user_scalars, user_duration = load_run_data(subdir, tags)
        except Exception as e:
            emit(f"  │  ✗ FAIL  Could not load TensorBoard data: {e}")
            overall_pass = False
            n_fail += 1
            emit(f"  └{'─'*60}")
            continue

        node_pass = True

        for tag in tags:
            ref_scalar  = ref_entry.get(tag)
            user_scalar = user_scalars.get(tag)
            label = f"  │    {tag:<14}"

            if ref_scalar is None and user_scalar is None:
                emit(f"{label}  —  not present in reference or user run (skipped)")
                continue
            if ref_scalar is None:
                emit(f"{label}  ⚠  not in golden — user has step={user_scalar['step']} value={user_scalar['value']:.6f} (not checked)")
                continue
            if user_scalar is None:
                emit(f"{label}  ✗ FAIL  missing in user run (present in golden)")
                node_pass = False
                continue

            passed, abs_diff, rel_pct = check_tolerance(user_scalar["value"], ref_scalar["value"], args.rel_tol)
            emit(f"{label}  {fmt_check(passed, user_scalar['value'], ref_scalar['value'], abs_diff, rel_pct)}")
            if not passed:
                node_pass = False

        if not args.no_time:
            label = f"  │    {'duration':<14}"
            ref_duration = ref_entry.get("duration_seconds")

            if ref_duration is None:
                if user_duration is not None:
                    emit(f"{label}  ⚠  not in golden — user ran for {fmt_duration(user_duration)} (not checked)")
                else:
                    emit(f"{label}  —  not available in golden or user run (skipped)")
            elif user_duration is None:
                emit(f"{label}  ⚠  could not determine duration from user run (skipped)")
            else:
                passed, _, rel_pct = check_tolerance(user_duration, ref_duration, args.time_tol)
                emit(f"{label}  {fmt_time_check(passed, user_duration, ref_duration, rel_pct)}")
                if not passed:
                    node_pass = False

        verdict = "✓ PASS" if node_pass else "✗ FAIL"
        emit(f"  │")
        emit(f"  │  Node result: {verdict}")
        emit(f"  └{'─'*60}")

        if node_pass:
            n_pass += 1
        else:
            n_fail += 1
            overall_pass = False

    emit()
    emit(sep)
    emit(f"  SUMMARY:  {n_pass} passed  |  {n_fail} failed  |  {n_skip} skipped")
    emit(f"  OVERALL:  {'✓ ALL PASSED' if overall_pass else '✗ FAILURES DETECTED'}")
    emit(sep)

    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nReport written → {args.report}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
