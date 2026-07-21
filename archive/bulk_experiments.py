"""
run_bulk_experiments.py
─────────────────────────────────────────────────────────────
Run pVisionTransformerTrainer on every combination of:
    • dataset            ∈ {mnist, fashionmnist, cifar10, cifar100, tinyimagenet}
    • ising_type         ∈ {LM_saliency_scores, diag_saliency_scores, no_saliency_scores}
    • val_split fraction ∈ {0.89, 0.85, 0.60}
30 independent seeds for each setting are executed.
Results are appended to bulk_results.csv in the working directory.
"""
import os
os.chdir('C:/My files/repos/Likelihood-guided-regularization-for-transformers')

import itertools, random, os, time, importlib, sys
import numpy as np
import pandas as pd
import torch
from argparse import Namespace
from tqdm.auto import tqdm  
from main.pVisionTransformer_Trainer import pVisionTransformerTrainer
from data_loader.cifar_downloader import *
from data_loader.mnist_fmnist_downloader import *
from data_loader.tinyimage_downloader import *


def ensure_data(dataset):

    """Skip downloading if known file/folder already exists."""
    expected_paths = {
        "mnist":        "./mnist/MNIST/raw/train-images-idx3-ubyte",
        "fashionmnist": "./fashionmnist/FashionMNIST/raw/train-images-idx3-ubyte",
        "cifar10":      "./cifar10/cifar-10-batches-py/data_batch_1",
        "cifar100":     "./cifar100/cifar-100-python/train",
        "tinyimagenet": "./tinyimagenet/tiny-imagenet-200/train",
    }
    
    if os.path.exists(expected_paths[dataset]):
        return  # Data is already present

    mod = {
        "mnist":        "data_loader.mnist_fmnist_downloader",
        "fashionmnist": "data_loader.mnist_fmnist_downloader",
        "cifar10":      "data_loader.cifar_downloader",
        "cifar100":     "data_loader.cifar_downloader",
        "tinyimagenet": "data_loader.tinyimage_downloader",
    }[dataset]

    importlib.import_module(mod)  # triggers download on import


# ─────────────────────────────────────────────────────────────
#  Fixed experiment grid
# ─────────────────────────────────────────────────────────────
DATASETS     = ["mnist", "fashionmnist", "cifar10", "cifar100", "tinyimagenet"]
ISING_TYPES  = ["LM_saliency_scores", "diag_saliency_scores", "no_saliency_scores"]
VAL_SPLITS   = [0.89, 0.80, 0.60]
RUNS_PER_CFG = 10                     # 30 seeds each

# Dataset-specific ViT hyper-parameters
# (feel free to tune patch_size / embed_dim / depth etc.)
MODEL_CFG = {
    "mnist"        : dict(img_size=28,  patch_size=7,  num_classes=10,  embed_dim=64, num_heads=4, depth=2),
    "fashionmnist" : dict(img_size=28,  patch_size=7,  num_classes=10,  embed_dim=64, num_heads=4, depth=2),
    "cifar10"      : dict(img_size=32,  patch_size=4,  num_classes=10,  embed_dim=128,num_heads=8, depth=6),
    "cifar100"     : dict(img_size=32,  patch_size=4,  num_classes=100, embed_dim=128,num_heads=8, depth=6),
    "tinyimagenet" : dict(img_size=64,  patch_size=4,  num_classes=200, embed_dim=256,num_heads=8, depth=8),
}

# Base directory for checkpoints of this sweep
CKPT_ROOT = "./checkpoints_bulk"

# CSV destination (append mode)
CSV_PATH  = "bulk_results_all_ising.csv"
if not os.path.exists(CSV_PATH):
    pd.DataFrame().to_csv(CSV_PATH, index=False)   # create headerless file

# ─────────────────────────────────────────────────────────────
def build_args(dataset, ising_type, val_split, seed):
    """Return a Namespace identical to your example `args`, but with
       dataset-specific and sweep-specific overrides."""
    base = dict(
        use_gpu           = torch.cuda.is_available(),
        device            = torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # ← added
        gpu               = 0,
        use_multi_gpu     = False,
        device_ids        = [0],
        num_models        = 1,
        dropout           = 0.0,
        batch_size        = 20,
        learning_rate     = 1e-3,
        kl_pen            = 1e-6,
        patience          = 100,
        lambda_weight1    = 1e-6,
        lambda_weight2    = 1e-6,
        train_epochs      = 25,   # ← match your example if intended
        ising_epochs      = 5,
        addtl_ft          = 25,
        ising_type        = ising_type,
        val_split         = val_split,
        test_split        = 0.10,
        ising_batch       = True,
        num_workers       = 0,
        dataset           = dataset,
        data_path         = f"./{dataset}",
        root_path         = ".",        # ← added
        checkpoints       = f"{CKPT_ROOT}/{dataset}/{ising_type}/val{val_split}",
        path              = ".",
        sim_seed          = seed,
        lradj             = "type2",    # ← added
    )
    base.update(MODEL_CFG[dataset])      # patch_size, embed_dim, etc.
    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return Namespace(**base)

# ─────────────────────────────────────────────────────────────
def run_one(cfg):
    dataset, ising_type, val_split, run_idx = cfg
    ensure_data(dataset)

    # ── ➊ reset PyTorch’s peak-tracker at the very start ───────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args     = build_args(dataset, ising_type, val_split, seed=run_idx)
    trainer  = pVisionTransformerTrainer(args)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ── ➋ query peak and current allocation just after training ───────
    peak_gib   = curr_gib = 0.0
    if torch.cuda.is_available():
        peak_gib  = torch.cuda.max_memory_allocated() / 1024**3
        curr_gib  = torch.cuda.memory_allocated()      / 1024**3

    stats = trainer.get_run_stats()
    flat  = {k: (v[0] if isinstance(v, list) else v) for k, v in stats.items()}
    
    # ── ➍ evaluate final model on test set ────────────────────────────
    eval_accuracies = trainer.evaluate(return_metrics=True)
    flat["final_test_accuracy"] = eval_accuracies[0] if isinstance(eval_accuracies, list) else eval_accuracies

    flat.update(dataset=dataset, ising_type=ising_type,
                val_split=val_split, run_idx=run_idx,
                elapsed_sec=round(elapsed, 2),
                peak_gpu_gib=round(peak_gib, 2),
                resid_gpu_gib=round(curr_gib, 2))

    # ── ➌ free objects & cache ────────────────────────────────────────
    del trainer, args
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        curr_after = torch.cuda.memory_allocated() / 1024**3
        flat["resid_gpu_gib_after_gc"] = round(curr_after, 2)

    MAX_ALLOWED = 6.0          # GiB, for an 8 GiB card
    if curr_after > MAX_ALLOWED:
        raise RuntimeError(f"Residual GPU memory {curr_after:.2f} GiB exceeds "
                        f"{MAX_ALLOWED} GiB – aborting sweep.")


    return flat

# ─────────────────────────────────────────────────────────────
def main():
    grid   = list(itertools.product(DATASETS, ISING_TYPES, VAL_SPLITS, range(RUNS_PER_CFG)))
    total  = len(grid)

    for cfg in tqdm(grid, total=total, desc="Bulk experiments", unit="run"):

        row    = run_one(cfg)
        # append row to CSV
        pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)

        # brief status text in bar
        tqdm.write(f"{cfg} | {row['elapsed_sec']} s  "
                f"peak {row['peak_gpu_gib']} GiB  "
                f"resid {row['resid_gpu_gib']} → {row['resid_gpu_gib_after_gc']} GiB")


if __name__ == "__main__":
    main()