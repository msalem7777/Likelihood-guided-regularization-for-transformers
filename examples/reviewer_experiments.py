"""Reviewer-focused experiment runner.

This script is intentionally separate from the paper's canonical reproduction runner.
It adds only the controlled studies requested during peer review: seed stability,
computational cost, warm-start sensitivity, posterior-scale / Ising-field sensitivity,
and quantitative calibration.

Examples
--------
python examples/reviewer_experiments.py --study cifar100_variance
python examples/reviewer_experiments.py --study efficiency
python examples/reviewer_experiments.py --study warmstart
python examples/reviewer_experiments.py --study sensitivity
python examples/reviewer_experiments.py --study sparse_vd

Smoke test (build cases only; no training):
python examples/reviewer_experiments.py --study sensitivity --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from argparse import Namespace

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from main.VisionTransformer_Trainer import VisionTransformerTrainer


MODEL_CFG = {
    "mnist": dict(img_size=28, patch_size=7, num_classes=10, embed_dim=32, num_heads=4, depth=2),
    "fashionmnist": dict(img_size=28, patch_size=7, num_classes=10, embed_dim=32, num_heads=4, depth=2),
    "cifar10": dict(img_size=32, patch_size=4, num_classes=10, embed_dim=64, num_heads=8, depth=2),
    "cifar100": dict(img_size=32, patch_size=4, num_classes=100, embed_dim=64, num_heads=8, depth=2),
}

DATASET_TOTALS = {
    "mnist": 60_000,
    "fashionmnist": 60_000,
    "cifar10": 50_000,
    "cifar100": 50_000,
}


@dataclass(frozen=True)
class RunSpec:
    study: str
    dataset: str
    train_samples: int
    seed: int
    method: str = "ising_lm"
    pilot_epochs: int = 5
    ising_epochs: int = 10
    fine_tune_epochs: int = 0
    dropconnect_delta: float = 0.5
    posterior_log_std: float = -4.0
    dropout: float = 0.0
    p_bayes: float = 0.0
    mc_samples: int = 128
    calibration_mc: int = 50
    calibration_bins: int = 15
    sparse_vd_threshold: float = 3.0
    sparse_vd_log_sigma_init: float = -5.0
    sparse_vd_kl_warmup_epochs: int = 15


def train_samples_to_val_split(dataset: str, train_samples: int, test_split: float = 0.10) -> float:
    """Convert an exact requested training count into the repo's val_split convention."""
    total = DATASET_TOTALS[dataset]
    train_fraction = train_samples / total
    val_split = 1.0 - test_split - train_fraction
    if not 0.0 <= val_split < 1.0:
        raise ValueError(
            f"Invalid train_samples={train_samples} for {dataset}; computed val_split={val_split}."
        )
    return val_split


def ensure_data(dataset: str) -> None:
    """Download the requested torchvision dataset if it is not already present."""
    from torchvision import datasets as tvd

    ctor = {
        "mnist": tvd.MNIST,
        "fashionmnist": tvd.FashionMNIST,
        "cifar10": tvd.CIFAR10,
        "cifar100": tvd.CIFAR100,
    }[dataset]
    ctor(root=f"./{dataset}", download=True)


def build_args(spec: RunSpec, output_root: Path) -> Namespace:
    """Translate one reviewer RunSpec into the trainer's existing Namespace API."""
    if spec.method == "ising_lm":
        ising_type = "LM_saliency_scores"
        ising_epochs = spec.ising_epochs
        dropout = 0.0
        p_bayes = 0.0
    elif spec.method == "ising_diag":
        ising_type = "diag_saliency_scores"
        ising_epochs = spec.ising_epochs
        dropout = 0.0
        p_bayes = 0.0
    elif spec.method == "ising_no_saliency":
        ising_type = "no_saliency_scores"
        ising_epochs = spec.ising_epochs
        dropout = 0.0
        p_bayes = 0.0
    elif spec.method == "dropout":
        ising_type = "no_saliency_scores"
        ising_epochs = 0
        dropout = spec.dropout
        p_bayes = 0.0
    elif spec.method == "dropconnect":
        ising_type = "no_saliency_scores"
        ising_epochs = 0
        dropout = 0.0
        p_bayes = spec.p_bayes
    elif spec.method == "sparse_vd":
        ising_type = "no_saliency_scores"
        ising_epochs = 0
        dropout = 0.0
        p_bayes = 0.0
    else:
        raise ValueError(f"Unsupported method: {spec.method}")

    val_split = train_samples_to_val_split(spec.dataset, spec.train_samples)
    run_tag = (
        f"seed{spec.seed}"
        f"_pilot{spec.pilot_epochs}"
        f"_ising{spec.ising_epochs}"
        f"_delta{spec.dropconnect_delta:g}"
        f"_logstd{spec.posterior_log_std:g}"
        f"_dropout{spec.dropout:g}"
        f"_pbayes{spec.p_bayes:g}"
        f"_svdthreshold{spec.sparse_vd_threshold:g}"
        f"_svdlogsig{spec.sparse_vd_log_sigma_init:g}"
    ).replace("-", "m").replace(".", "p")
    ckpt = output_root / "checkpoints" / spec.study / spec.dataset / spec.method / run_tag

    base = dict(
        use_gpu=torch.cuda.is_available(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gpu=0,
        use_multi_gpu=False,
        device_ids=[0],
        num_models=1,
        method=spec.method,
        dropout=dropout,
        dropconnect_delta=spec.dropconnect_delta,
        p_bayes=p_bayes,
        posterior_log_std=spec.posterior_log_std,
        batch_size=20,
        learning_rate=1e-3,
        kl_pen=1e-6,
        patience=100,
        lambda_weight1=1e-6,
        lambda_weight2=1e-6,
        train_epochs=spec.pilot_epochs,
        ising_epochs=ising_epochs,
        addtl_ft=spec.fine_tune_epochs,
        ising_type=ising_type,
        disable_early_stopping=True,
        drop_thresh=0.5,
        val_split=val_split,
        test_split=0.10,
        ising_batch=False,
        num_workers=4,
        dataset=spec.dataset,
        data_path=f"./{spec.dataset}",
        root_path=".",
        checkpoints=str(ckpt),
        path=".",
        sim_seed=spec.seed,
        split_seed=spec.seed,
        lradj="type2",
        mc_samples=spec.mc_samples,
        hessian_block_size=256,
        sparse_vd_threshold=spec.sparse_vd_threshold,
        sparse_vd_log_sigma_init=spec.sparse_vd_log_sigma_init,
        sparse_vd_kl_warmup_epochs=spec.sparse_vd_kl_warmup_epochs,
    )
    base.update(MODEL_CFG[spec.dataset])
    return Namespace(**base)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str | None:
    """Return the exact git commit for reproducibility when running from a clone."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware_metadata() -> dict[str, object]:
    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    else:
        meta["gpu_name"] = None
        meta["gpu_total_gib"] = 0.0
    return meta


def build_study(study: str) -> list[RunSpec]:
    """Return the prespecified minimal reviewer run matrix for one study."""
    specs: list[RunSpec] = []

    if study == "cifar100_variance":
        # Directly addresses the anomalous CIFAR-100 n=15,000 variance noted by Referee 1.
        for seed in range(10):
            for delta in (0.1, 0.5):
                specs.append(RunSpec(study, "cifar100", 15_000, seed, dropconnect_delta=delta))

    elif study == "efficiency":
        # One easy and one hard setting; matched seeds. Exact diagonal Hessian is restricted
        # to MNIST because its purpose here is cost quantification, not a full accuracy sweep.
        for seed in range(3):
            for dataset, n_train in (("mnist", 6_000), ("cifar100", 15_000)):
                specs.extend([
                    RunSpec(study, dataset, n_train, seed, method="ising_lm"),
                    RunSpec(study, dataset, n_train, seed, method="ising_no_saliency"),
                    RunSpec(study, dataset, n_train, seed, method="dropout", dropout=0.1),
                    RunSpec(study, dataset, n_train, seed, method="dropconnect", p_bayes=0.1),
                ])
            specs.append(RunSpec(study, "mnist", 6_000, seed, method="ising_diag"))

    elif study == "warmstart":
        # Hold total pilot+Ising epochs at 15 so the stopping-point comparison does not
        # simply reward longer total training.
        for seed in range(5):
            for pilot_epochs in (1, 5, 10):
                specs.append(RunSpec(
                    study, "cifar100", 15_000, seed,
                    pilot_epochs=pilot_epochs,
                    ising_epochs=15 - pilot_epochs,
                ))

    elif study == "sensitivity":
        # One-factor-at-a-time sensitivity around the reported default (delta=.5, log sigma=-4).
        configs = [
            (0.1, -4.0),
            (0.25, -4.0),
            (0.5, -4.0),
            (0.75, -4.0),
            (0.5, -5.0),
            (0.5, -3.0),
        ]
        for seed in range(3):
            for delta, log_std in configs:
                specs.append(RunSpec(
                    study, "cifar100", 15_000, seed,
                    dropconnect_delta=delta,
                    posterior_log_std=log_std,
                ))

    elif study == "sparse_vd":
        # Minimal stronger Bayesian baseline: the two reviewer-table settings
        # with three matched seeds and the same 15-epoch training allocation.
        for dataset, n_train in (("mnist", 6_000), ("cifar100", 15_000)):
            for seed in range(3):
                specs.append(RunSpec(
                    study,
                    dataset,
                    n_train,
                    seed,
                    method="sparse_vd",
                    pilot_epochs=15,
                    ising_epochs=0,
                    sparse_vd_kl_warmup_epochs=15,
                ))
    else:
        raise ValueError(f"Unknown study: {study}")

    return specs


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one(spec: RunSpec, output_root: Path) -> dict[str, object]:
    ensure_data(spec.dataset)
    seed_everything(spec.seed)
    args = build_args(spec, output_root)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = VisionTransformerTrainer(args)

    train_t0 = time.perf_counter()
    trainer.train()
    train_sec = time.perf_counter() - train_t0

    deterministic_t0 = time.perf_counter()
    final_acc = trainer.evaluate(return_metrics=True)
    deterministic_test_sec = time.perf_counter() - deterministic_t0

    calibration_t0 = time.perf_counter()
    calibration = trainer.evaluate_calibration(
        n_mc=spec.calibration_mc,
        n_bins=spec.calibration_bins,
    )[0]
    mc_test_sec = time.perf_counter() - calibration_t0

    stats = trainer.get_run_stats()
    row = {**asdict(spec)}
    row.update({k: (v[0] if isinstance(v, list) else v) for k, v in stats.items()})
    row.update({
        "final_test_accuracy": final_acc[0] if isinstance(final_acc, list) else final_acc,
        "train_wall_sec": train_sec,
        "deterministic_test_sec": deterministic_test_sec,
        "mc_test_sec": mc_test_sec,
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "brier": calibration["brier"],
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        **hardware_metadata(),
    })
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        required=True,
        choices=("cifar100_variance", "efficiency", "warmstart", "sensitivity", "sparse_vd"),
    )
    parser.add_argument("--output-root", default="reviewer_results")
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help=(
            "Run exactly one zero-based matrix entry. This is intended for SLURM job arrays; "
            "SLURM_ARRAY_TASK_ID can be passed directly."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the selected run matrix without training.")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    all_specs = build_study(cli.study)
    output_root = Path(cli.output_root)

    if cli.run_index is not None:
        if not 0 <= cli.run_index < len(all_specs):
            raise SystemExit(
                f"--run-index must be in [0, {len(all_specs) - 1}] for study {cli.study}; "
                f"received {cli.run_index}."
            )
        selected = [(cli.run_index, all_specs[cli.run_index])]
    else:
        selected = list(enumerate(all_specs))

    if cli.dry_run:
        print(json.dumps([asdict(spec) for _, spec in selected], indent=2))
        print(f"Selected runs: {len(selected)} / {len(all_specs)}")
        return

    for matrix_index, spec in selected:
        # Each matrix entry writes a separate CSV. This makes SLURM-array execution
        # race-free and preserves partial results if another task fails.
        csv_path = output_root / cli.study / f"run_{matrix_index:03d}.csv"
        print(f"[matrix {matrix_index}/{len(all_specs) - 1}] {spec}")
        row = run_one(spec, output_root)
        row = {"matrix_index": matrix_index, **row}
        append_csv(csv_path, row)
        print(
            f"  accuracy={row['final_test_accuracy']} | ECE={row['ece']:.4f} | "
            f"train={row['train_wall_sec']:.1f}s | peak={row['peak_gpu_gib']:.2f} GiB"
        )


if __name__ == "__main__":
    main()
