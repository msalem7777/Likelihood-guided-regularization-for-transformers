"""Merge race-free per-run reviewer CSV files into one CSV per study."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reviewer_results")
    parser.add_argument(
        "--study",
        choices=("cifar100_variance", "efficiency", "warmstart", "sensitivity"),
        required=True,
    )
    args = parser.parse_args()

    study_dir = Path(args.root) / args.study
    files = sorted(study_dir.glob("run_*.csv"))
    if not files:
        raise SystemExit(f"No run CSV files found in {study_dir}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True).sort_values("matrix_index")
    out = Path(args.root) / f"{args.study}.csv"
    merged.to_csv(out, index=False)
    print(f"Merged {len(files)} files -> {out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
