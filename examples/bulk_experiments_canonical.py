"""
bulk_experiments_canonical.py
─────────────────────────────────────────────────────────────
Canonical sweep runner (post-2026-07 pipeline: seeded splits, patched trainer).
All experiment settings live in the CONFIG block below — edit there, nothing else.
Each (dataset, ising_type, val_split, seed) cell appends one row to CSV_PATH.
Checkpoints are seed-scoped (no cross-seed overwrites).
Supersedes the bulk_experiments_* variant family, which is kept frozen for
provenance of archived results.
"""
import itertools, random, os, sys, time, importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from argparse import Namespace
from tqdm.auto import tqdm
from main.VisionTransformer_Trainer import VisionTransformerTrainer
from data_loader.cifar_downloader import *
from data_loader.mnist_fmnist_downloader import *

# ─────────────────────────────────────────────────────────────
#  CONFIG — the only block you edit per experiment
# ─────────────────────────────────────────────────────────────
DATASETS     = ["mnist"]
ISING_TYPES  = ["LM_saliency_scores"]     # LM_saliency_scores | diag_saliency_scores | no_saliency_scores
VAL_SPLITS   = [0.895]
RUNS_PER_CFG = 1                          # seeds 0..N-1

TRAIN_EPOCHS = 1
ISING_EPOCHS = 5
ADDTL_FT     = 0
DROPOUT      = 0.0
DROPCONN     = 0.1                        # dropconnect_delta (external field)
P_BAYES      = 0.0                        # Bayesian layer drop prob
NUM_WORKERS  = 4                          # 0 in notebooks; 4 local; 8 on ARC
SPLIT_SEED_FOLLOWS_RUN = True             # True: partition varies per seed; False: fixed 42

MODEL_CFG = {
    "mnist"        : dict(img_size=28, patch_size=7, num_classes=10,  embed_dim=32, num_heads=4, depth=2),
    "fashionmnist" : dict(img_size=28, patch_size=7, num_classes=10,  embed_dim=32, num_heads=4, depth=2),
    "cifar10"      : dict(img_size=32, patch_size=4, num_classes=10,  embed_dim=64, num_heads=8, depth=2),
    "cifar100"     : dict(img_size=32, patch_size=4, num_classes=100, embed_dim=64, num_heads=8, depth=2),
}

TAG = (f"{'-'.join(ISING_TYPES)}_train{TRAIN_EPOCHS}_ising{ISING_EPOCHS}_ft{ADDTL_FT}"
       f"_do{DROPOUT}_dc{DROPCONN}_bdo{P_BAYES}")
CKPT_ROOT = f"./checkpoints_{TAG}"
CSV_PATH  = f"bulk_results_{TAG}.csv"
# ─────────────────────────────────────────────────────────────

def ensure_data(dataset):
    """Skip downloading if known file/folder already exists."""
    expected_paths = {
        "mnist":        "./mnist/MNIST/raw/train-images-idx3-ubyte",
        "fashionmnist": "./fashionmnist/FashionMNIST/raw/train-images-idx3-ubyte",
        "cifar10":      "./cifar10/cifar-10-batches-py/data_batch_1",
        "cifar100":     "./cifar100/cifar-100-python/train",
    }
    if os.path.exists(expected_paths[dataset]):
        return
    mod = {
        "mnist":        "data_loader.mnist_fmnist_downloader",
        "fashionmnist": "data_loader.mnist_fmnist_downloader",
        "cifar10":      "data_loader.cifar_downloader",
        "cifar100":     "data_loader.cifar_downloader",
    }[dataset]
    importlib.import_module(mod)

def build_args(dataset, ising_type, val_split, seed):
    base = dict(
        use_gpu           = torch.cuda.is_available(),
        device            = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gpu               = 0,
        use_multi_gpu     = False,
        device_ids        = [0],
        num_models        = 1,
        dropout           = DROPOUT,
        dropconnect_delta = DROPCONN,
        p_bayes           = P_BAYES,
        batch_size        = 20,
        learning_rate     = 1e-3,
        kl_pen            = 1e-6,
        patience          = 100,
        lambda_weight1    = 1e-6,
        lambda_weight2    = 1e-6,
        train_epochs      = TRAIN_EPOCHS,
        ising_epochs      = ISING_EPOCHS,
        addtl_ft          = ADDTL_FT,
        ising_type        = ising_type,
        disable_early_stopping = True,
        drop_thresh       = 0.5,
        val_split         = val_split,
        test_split        = 0.10,
        ising_batch       = False,
        num_workers       = NUM_WORKERS,
        dataset           = dataset,
        data_path         = f"./{dataset}",
        root_path         = ".",
        # seed in the checkpoint path: prevents cross-seed overwrite/cross-load
        checkpoints       = f"{CKPT_ROOT}/{dataset}/{ising_type}/val{val_split}/seed{seed}",
        path              = ".",
        sim_seed          = seed,
        split_seed        = seed if SPLIT_SEED_FOLLOWS_RUN else 42,
        lradj             = "type2",
        mc_samples        = 128,
        hessian_block_size = 1024,
    )
    base.update(MODEL_CFG[dataset])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return Namespace(**base)

def run_one(cfg):
    dataset, ising_type, val_split, run_idx = cfg
    ensure_data(dataset)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args    = build_args(dataset, ising_type, val_split, seed=run_idx)
    trainer = VisionTransformerTrainer(args)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_gib = curr_gib = 0.0
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        curr_gib = torch.cuda.memory_allocated()     / 1024**3

    stats = trainer.get_run_stats()
    flat  = {k: (v[0] if isinstance(v, list) else v) for k, v in stats.items()}

    eval_accuracies = trainer.evaluate(return_metrics=True)
    flat["final_test_accuracy"] = eval_accuracies[0] if isinstance(eval_accuracies, list) else eval_accuracies

    flat.update(dataset=dataset, ising_type=ising_type,
                val_split=val_split, run_idx=run_idx,
                split_seed=args.split_seed,
                elapsed_sec=round(elapsed, 2),
                peak_gpu_gib=round(peak_gib, 2),
                resid_gpu_gib=round(curr_gib, 2))

    del trainer, args
    import gc
    gc.collect()
    curr_after = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        curr_after = torch.cuda.memory_allocated() / 1024**3
    flat["resid_gpu_gib_after_gc"] = round(curr_after, 2)

    MAX_ALLOWED = 4.0
    if curr_after > MAX_ALLOWED:
        raise RuntimeError(f"Residual GPU memory {curr_after:.2f} GiB exceeds "
                           f"{MAX_ALLOWED} GiB - aborting sweep.")
    return flat

def main():
    if not os.path.exists(CSV_PATH):
        pd.DataFrame().to_csv(CSV_PATH, index=False)

    grid = list(itertools.product(DATASETS, ISING_TYPES, VAL_SPLITS, range(RUNS_PER_CFG)))

    for cfg in tqdm(grid, total=len(grid), desc="Bulk experiments", unit="run"):
        row = run_one(cfg)
        pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
        tqdm.write(f"{cfg} | {row['elapsed_sec']} s  "
                   f"peak {row['peak_gpu_gib']} GiB  "
                   f"resid {row['resid_gpu_gib']} -> {row['resid_gpu_gib_after_gc']} GiB")

    # Archive checkpoints for this sweep (was previously outside main)
    import shutil, datetime
    if os.path.exists(CKPT_ROOT):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(CKPT_ROOT, f"{CKPT_ROOT}_archived_{ts}")

if __name__ == "__main__":
    main()