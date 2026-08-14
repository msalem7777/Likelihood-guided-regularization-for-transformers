"""Smoke-test the 240-run missing-dataset reviewer extension.

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
    assert len(specs) == 240, f"expected 240 runs, found {len(specs)}"

    expected = []
    settings = (
        ("mnist", 300),
        ("mnist", 18_000),
        ("fashionmnist", 300),
        ("fashionmnist", 6_000),
        ("fashionmnist", 18_000),
        ("cifar10", 250),
        ("cifar10", 5_000),
        ("cifar10", 15_000),
        ("cifar100", 250),
        ("cifar100", 5_000),
    )
    for dataset, n_train in settings:
        for seed in range(3):
            for hyperparameter in (0.1, 0.5):
                expected.extend([
                    (
                        dataset,
                        n_train,
                        seed,
                        "ising_lm",
                        hyperparameter,
                    ),
                    (
                        dataset,
                        n_train,
                        seed,
                        "ising_no_saliency",
                        hyperparameter,
                    ),
                    (
                        dataset,
                        n_train,
                        seed,
                        "dropout",
                        hyperparameter,
                    ),
                    (
                        dataset,
                        n_train,
                        seed,
                        "dropconnect",
                        hyperparameter,
                    ),
                ])

    observed = [
        (
            spec.dataset,
            spec.train_samples,
            spec.seed,
            spec.method,
            (
                spec.dropout
                if spec.method == "dropout"
                else spec.p_bayes
                if spec.method == "dropconnect"
                else spec.dropconnect_delta
            ),
        )
        for spec in specs
    ]
    assert observed == expected

    for spec in specs:
        args = build_args(spec, Path("reviewer_results"))

        if spec.method.startswith("ising_"):
            # Paper protocol: one pilot epoch followed by 70 Ising epochs.
            assert args.train_epochs == spec.pilot_epochs == 1
            assert args.ising_epochs == spec.ising_epochs == 70
        else:
            # Matched baselines receive the same total 71-epoch budget.
            assert args.train_epochs == 71
            assert args.ising_epochs == 0

        assert args.train_epochs + args.ising_epochs + args.addtl_ft == 71
        assert args.batch_size == 20
        assert args.learning_rate == 1e-3
        if spec.method == "dropout":
            assert args.dropout in (0.1, 0.5)
        else:
            assert args.dropout == 0.0
        if spec.method == "dropconnect":
            assert args.p_bayes in (0.1, 0.5)
        else:
            assert args.p_bayes == 0.0

    print("MISSING-DATASET REVIEWER MATRIX SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
