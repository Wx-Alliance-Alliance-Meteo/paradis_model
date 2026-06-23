#!/usr/bin/env python3
"""
Extract last train_loss and val_loss from TensorBoard event files
across benchmark-025_*node subdirectories.

Usage:
    python extract_benchmark_losses.py /path/to/benchmark/dir
    python extract_benchmark_losses.py /path/to/benchmark/dir --csv results.csv
"""

import argparse
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def parse_node_count(dirname: str) -> int:
    """Extract the numeric node count from a directory name like benchmark-025_3node."""
    m = re.search(r"_(\d+)node$", dirname)
    return int(m.group(1)) if m else -1


def load_last_scalars(log_dir: Path, tags: list[str]) -> tuple[dict[str, tuple[int, float] | None], float | None]:
    """
    Load an EventAccumulator from log_dir and return:
      - a dict of tag -> last (step, value), or None if the tag is absent
      - training duration in seconds (wall_time of last event minus first event),
        or None if it cannot be determined
    """
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    results = {}
    for tag in tags:
        if tag not in available:
            results[tag] = None
        else:
            events = ea.Scalars(tag)
            last = events[-1]
            results[tag] = (last.step, last.value)

    all_wall_times: list[float] = []
    for tag in available:
        events = ea.Scalars(tag)
        if events:
            all_wall_times.append(events[0].wall_time)
            all_wall_times.append(events[-1].wall_time)

    duration = (max(all_wall_times) - min(all_wall_times)) if len(all_wall_times) >= 2 else None

    return results, duration


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds as H:MM:SS, or 'N/A' if None."""
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Extract last train/val loss from benchmark dirs.")
    parser.add_argument("dirpath", type=Path, help="Root directory containing benchmark-025_*node subdirs")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write CSV output")
    args = parser.parse_args()

    root = args.dirpath
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    pattern = "benchmark-025_*node"
    subdirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"benchmark-025_\d+node$", d.name)],
        key=lambda d: parse_node_count(d.name),
    )

    if not subdirs:
        print(f"No directories matching '{pattern}' found under {root}", file=sys.stderr)
        sys.exit(1)

    tags = ["train_loss", "val_loss"]
    rows = []

    print(f"Found {len(subdirs)} benchmark directories\n")
    print(
        f"{'Directory':<30}  {'Nodes':>5}  {'train_loss step':>15}  {'train_loss':>12}"
        f"  {'val_loss step':>13}  {'val_loss':>10}  {'duration':>10}"
    )
    print("-" * 110)

    for subdir in subdirs:
        nodes = parse_node_count(subdir.name)
        try:
            results, duration = load_last_scalars(subdir, tags)
        except Exception as e:
            print(f"  {subdir.name:<28}  {nodes:>5}  ERROR: {e}")
            continue

        train = results["train_loss"]
        val   = results["val_loss"]

        train_step  = train[0] if train else "N/A"
        train_value = f"{train[1]:.6f}" if train else "N/A"
        val_step    = val[0]   if val   else "N/A"
        val_value   = f"{val[1]:.6f}"   if val   else "N/A"
        dur_str     = format_duration(duration)

        print(
            f"  {subdir.name:<28}  {nodes:>5}  "
            f"{str(train_step):>15}  {train_value:>12}  "
            f"{str(val_step):>13}  {val_value:>10}  "
            f"{dur_str:>10}"
        )

        rows.append({
            "directory":        subdir.name,
            "nodes":            nodes,
            "train_loss_step":  train[0] if train else None,
            "train_loss_value": train[1] if train else None,
            "val_loss_step":    val[0]   if val   else None,
            "val_loss_value":   val[1]   if val   else None,
            "duration_seconds": round(duration) if duration is not None else None,
        })

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
