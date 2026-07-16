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
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
from tqdm.auto import tqdm  
from scipy.stats import entropy
from main.VisionTransformer_Trainer import VisionTransformerTrainer
from data_loader.cifar_downloader import *
from data_loader.mnist_fmnist_downloader import *


def ensure_data(dataset):

    """Skip downloading if known file/folder already exists."""
    expected_paths = {
        "mnist":        "./mnist/MNIST/raw/train-images-idx3-ubyte",
        "fashionmnist": "./fashionmnist/FashionMNIST/raw/train-images-idx3-ubyte",
        "cifar10":      "./cifar10/cifar-10-batches-py/data_batch_1",
        "cifar100":     "./cifar100/cifar-100-python/train",
    }
    
    if os.path.exists(expected_paths[dataset]):
        return  # Data is already present

    mod = {
        "mnist":        "data_loader.mnist_fmnist_downloader",
        "fashionmnist": "data_loader.mnist_fmnist_downloader",
        "cifar10":      "data_loader.cifar_downloader",
        "cifar100":     "data_loader.cifar_downloader",
    }[dataset]

    importlib.import_module(mod)  # triggers download on import


# ─────────────────────────────────────────────────────────────
#  Fixed experiment grid
# ─────────────────────────────────────────────────────────────
DATASETS     = ["mnist", "fashionmnist", "cifar10", "cifar100"]
# ISING_TYPES  = ["LM_saliency_scores", "diag_saliency_scores", "no_saliency_scores"]
ISING_TYPES  = ["LM_saliency_scores"]
VAL_SPLITS   = [0.895, 0.8, 0.60]
RUNS_PER_CFG = 10                     # 30 seeds each

# Dataset-specific ViT hyper-parameters
# (feel free to tune patch_size / embed_dim / depth etc.)
MODEL_CFG = {
    "mnist"        : dict(img_size=28,  patch_size=7,  num_classes=10,  embed_dim=64, num_heads=8, depth=2),
    "fashionmnist" : dict(img_size=28,  patch_size=7,  num_classes=10,  embed_dim=64, num_heads=8, depth=2),
    "cifar10"      : dict(img_size=32,  patch_size=4,  num_classes=10,  embed_dim=128, num_heads=16, depth=2),
    "cifar100"     : dict(img_size=32,  patch_size=4,  num_classes=100, embed_dim=128, num_heads=16, depth=2),
}

# Optional: override these if needed
TRAIN_EPOCHS = 71
ISING_EPOCHS = 0
ADDTL_FT     = 0
DROPOUT      = 0.0
DROPCONN     = 0.0
P_BAYES      = 0.5

# Create configuration tag
epoch_desc = f"train{TRAIN_EPOCHS}_ising{ISING_EPOCHS}_ft{ADDTL_FT}"
reg_params = f"do{DROPOUT}_dc{DROPCONN}_bdo_{P_BAYES}"

# Root directory and CSV path
TAG = f"{ISING_TYPES[0]}_{epoch_desc}_{reg_params}"

CKPT_ROOT = f"./checkpoints_bulk_{TAG}"
CSV_PATH  = f"bulk_results_{TAG}.csv"

if not os.path.exists(CSV_PATH):
    pd.DataFrame().to_csv(CSV_PATH, index=False)  # Create empty CSV if needed

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
        dropout           = DROPOUT,
        p_bayes           = P_BAYES,
        dropconnect_delta = DROPCONN,
        batch_size        = 20,
        learning_rate     = 1e-3,
        kl_pen            = 1e-6,
        patience          = 100,
        lambda_weight1    = 1e-6,
        lambda_weight2    = 1e-6,
        train_epochs      = TRAIN_EPOCHS,   # ← match your example if intended
        ising_epochs      = ISING_EPOCHS,
        addtl_ft          = ADDTL_FT,
        ising_type        = ising_type,
        disable_early_stopping = True,
        drop_thresh       = 0.5,
        val_split         = val_split,
        test_split        = 0.10,
        ising_batch       = False,
        num_workers       = 0,
        dataset           = dataset,
        data_path         = f"./{dataset}",
        root_path         = ".",        # ← added
        checkpoints       = f"{CKPT_ROOT}/{dataset}/{ising_type}/val{val_split}",
        path              = ".",
        sim_seed          = seed,
        lradj             = "type4",    # ← added
        mc_samples        = 128
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
    trainer  = VisionTransformerTrainer(args)

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

    # ── NEW: Save prediction DataFrames for each model ────────────────
    # Standard deterministic prediction
    pred_dfs = trainer.predict_with_probs()

    for model_idx, pred_df in enumerate(pred_dfs):
        pred_df.to_csv(os.path.join(args.checkpoints, f"pred_probs_model_{model_idx}_run{run_idx}.csv"), index=False)
        num_classes = pred_df['true_label'].max() + 1

        # Get probabilities and predicted classes
        pred_probs = pred_df[[f'prob_class_{k}' for k in range(num_classes)]].values
        pred_classes = np.argmax(pred_probs, axis=1)
        top_pred_probs = pred_probs[np.arange(len(pred_probs)), pred_classes]
        true_labels = pred_df['true_label'].values
        correct_mask = pred_classes == true_labels

        # Split into correct and incorrect
        correct_probs = top_pred_probs[correct_mask]
        incorrect_probs = top_pred_probs[~correct_mask]

        # Side-by-side histogram plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        axes[0].hist(correct_probs, bins=30, color='green', edgecolor='k', alpha=0.7)
        axes[0].set_title('Predicted class probability\n(Correct predictions)')
        axes[0].set_xlabel('Probability')
        axes[0].set_ylabel('Frequency')
        axes[1].hist(incorrect_probs, bins=30, color='red', edgecolor='k', alpha=0.7)
        axes[1].set_title('Predicted class probability\n(Incorrect predictions)')
        axes[1].set_xlabel('Probability')
        plt.suptitle(f"Predicted Class Probabilities\n{dataset} Run: {run_idx} Model: {model_idx}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(args.checkpoints, f"hist_pred_prob_split_model_{model_idx}_run{run_idx}.png"))
        plt.close()

        # (1) Histogram: predicted probability of top predicted class per sample
        plt.figure(figsize=(7,5))
        plt.hist(top_pred_probs, bins=30, color='skyblue', edgecolor='k')
        plt.title(f'Hist: Predicted Probability (Top Class)\n{dataset} Run: {run_idx}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoints, f"hist_pred_prob_model_{model_idx}_run{run_idx}.png"))
        plt.close()

        # (2) Histogram: entropy of predicted distribution per sample
        from scipy.stats import entropy
        entropies = entropy(pred_probs.T, base=2)
        plt.figure(figsize=(7,5))
        plt.hist(entropies, bins=30, color='orange', edgecolor='k')
        plt.title(f'Hist: Prediction Entropy\n{dataset} Run: {run_idx}')
        plt.xlabel('Entropy (bits)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoints, f"hist_entropy_model_{model_idx}_run{run_idx}.png"))
        plt.close()

        # (6) Scatter: Entropy vs Correctness
        correct = (pred_classes == true_labels).astype(int)
        plt.figure(figsize=(7,5))
        plt.scatter(entropies, correct, alpha=0.05, s=10)
        plt.title(f'Scatter: Entropy vs Correctness\n{dataset} Run: {run_idx}')
        plt.xlabel('Entropy (bits)')
        plt.ylabel('Correct (1) / Incorrect (0)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoints, f"scatter_entropy_correct_model_{model_idx}_run{run_idx}.png"))
        plt.close()

        n_show = 10  # Number of random samples to visualize

        # Group samples by true class, then stack their probabilities
        prob_rows = []
        ytick_pos = []
        ytick_labels = []
        row_idx = 0

        for c in range(num_classes):
            class_df = pred_df[pred_df['true_label'] == c]
            # Stack deterministic softmax probabilities for all samples in this class
            class_probs = np.stack([class_df[f'prob_class_{k}'] for k in range(num_classes)], axis=1)
            prob_rows.append(class_probs)
            # For y-ticks
            center = row_idx + class_probs.shape[0] // 2
            ytick_pos.append(center)
            ytick_labels.append(str(c))
            row_idx += class_probs.shape[0]

        prob_matrix = np.vstack(prob_rows)

        plt.figure(figsize=(12, 8))
        plt.imshow(prob_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label="Deterministic Softmax Probability")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class Label")
        plt.title(f"Deterministic Softmax Probability Heatmap\nDataset: {dataset} Run: {run_idx}")
        plt.xticks(np.arange(num_classes), [str(i) for i in range(num_classes)])
        plt.yticks(ytick_pos, ytick_labels)

        # Draw horizontal lines between class blocks
        row = 0
        for c in range(num_classes - 1):
            row += prob_rows[c].shape[0]
            plt.axhline(row - 0.5, color='white', linestyle='--', linewidth=1)

        plot_path = os.path.join(args.checkpoints, f"heatmap_model_{model_idx}_run{run_idx}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # MC posterior prediction
    mc_n = 100  # You can parameterize this if desired
    mc_quantiles = 10  # Deciles
    mc_pred_dfs = trainer.predict_mc_posterior_weight_sampling_quantiles(n_mc=mc_n, n_quantiles=mc_quantiles)
    # >>> CHANGES START
    for model_idx, df in enumerate(mc_pred_dfs):
        df.to_csv(os.path.join(args.checkpoints, f"mc_posterior_quantiles_model_{model_idx}_run{run_idx}.csv"), index=False)
        num_classes = df['true_label'].max() + 1  # <--- fix scope

        n_show = 10  # Number of random samples to visualize
        sample_indices = np.random.choice(len(df), size=n_show, replace=False)
        classes = range(num_classes)

        for i in sample_indices:
            quant_10 = [df.iloc[i][f'prob_class_{c}_quantile_0.10'] for c in classes]
            quant_50 = [df.iloc[i][f'prob_class_{c}_quantile_0.50'] for c in classes]
            quant_90 = [df.iloc[i][f'prob_class_{c}_quantile_0.90'] for c in classes]
            true_label = df.iloc[i]['true_label']

            # Error bars: from 10th to 90th quantile, centered at 50th
            lower = np.array(quant_50) - np.array(quant_10)
            upper = np.array(quant_90) - np.array(quant_50)

            plt.figure(figsize=(8,4))
            plt.errorbar(classes, quant_50, yerr=[lower, upper], fmt='o', capsize=5, color='blue')
            plt.title(f'MC Posterior Quantiles (Sample {i})\nTrue label: {true_label} | {dataset} Run: {run_idx}')
            plt.xlabel('Class')
            plt.ylabel('Predicted Probability')
            plt.xticks(classes)
            plt.ylim(0,1)
            plt.tight_layout()
            plt.savefig(os.path.join(args.checkpoints, f"mc_quantile_errorbars_sample_{i}_model_{model_idx}_run{run_idx}.png"))
            plt.close()
    # <<< CHANGES END

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

    MAX_ALLOWED = 4.0          # GiB, for an 8 GiB card
    if curr_after > MAX_ALLOWED:
        raise RuntimeError(f"Residual GPU memory {curr_after:.2f} GiB exceeds "
                        f"{MAX_ALLOWED} GiB – aborting sweep.")


    return flat

# ─────────────────────────────────────────────────────────────
def main():
    grid   = list(itertools.product(DATASETS, ISING_TYPES, VAL_SPLITS, range(RUNS_PER_CFG)))
    total  = len(grid)

    # Load completed runs from CSV, if it exists
    completed = set()
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 3:
        df_done = pd.read_csv(CSV_PATH, header=None)
        for _, row in df_done.iterrows():
            completed.add((row[0], row[18], float(row[19]), int(row[20])))
    else:
        # CSV does not exist or is empty: nothing is completed yet
        pass

    for cfg in tqdm(grid, total=total, desc="Bulk experiments", unit="run"):
        if cfg in completed:
            tqdm.write(f"Skipping completed: {cfg}")
            continue

        row = run_one(cfg)
        pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
        tqdm.write(f"{cfg} | {row['elapsed_sec']} s  "
                   f"peak {row['peak_gpu_gib']} GiB  "
                   f"resid {row['resid_gpu_gib']} → {row['resid_gpu_gib_after_gc']} GiB")


if __name__ == "__main__":
    main()
    import shutil, datetime

    if os.path.exists(CKPT_ROOT):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(CKPT_ROOT, f"{CKPT_ROOT}_archived_{ts}")
#%%