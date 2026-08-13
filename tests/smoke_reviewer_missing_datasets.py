"""Smoke-test the 24-run missing-dataset reviewer extension.

Run from the repository root:

    python tests/smoke_reviewer_missing_datasets.py

This test builds configurations only. It does not download data or train a model.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.reviewer_experiments import build_args, build_study


def main() -> None:
    specs = build_study("efficiency_missing_datasets")
    assert len(specs) == 24, f"expected 24 runs, found {len(specs)}"

    expected = []
    for dataset, n_train in (("fashionmnist", 6_000), ("cifar10", 15_000)):
        for seed in range(3):
            expected.extend([
                (dataset, n_train, seed, "ising_lm"),
                (dataset, n_train, seed, "ising_no_saliency"),
                (dataset, n_train, seed, "dropout"),
                (dataset, n_train, seed, "dropconnect"),
            ])

    observed = [
        (spec.dataset, spec.train_samples, spec.seed, spec.method)
        for spec in specs
    ]
    assert observed == expected

    for spec in specs:
        args = build_args(spec, Path("reviewer_results"))
        assert args.train_epochs == 5
        assert args.ising_epochs == (10 if spec.method.startswith("ising_") else 0)
        assert args.batch_size == 20
        assert args.learning_rate == 1e-3
        assert args.dropout == (0.1 if spec.method == "dropout" else 0.0)
        assert args.p_bayes == (0.1 if spec.method == "dropconnect" else 0.0)

    print("MISSING-DATASET REVIEWER MATRIX SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
