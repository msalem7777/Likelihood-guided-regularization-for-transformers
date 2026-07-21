#%%
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
#%%
# ---- Set these variables directly ----
ckpt_root = Path(".")                          # run this file from the repo root
out_dir   = Path("./examples/analysis_results")
pattern = "*.csv"                        # <-- adjust if needed

out_dir.mkdir(exist_ok=True, parents=True)

def compute_macro_fpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr = fp / (fp + tn)
        fpr = np.nan_to_num(fpr, nan=0.0)
    return float(np.mean(fpr))

def process_pred_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    prob_cols = sorted([c for c in df.columns if c.startswith("mc_prob_mean")],
                       key=lambda x: int(x.split("_")[-1]))
    if prob_cols:
        pred_probs = df[prob_cols].values  # shape: (n_samples, n_classes)

    if df.shape[1] < 2:
        raise ValueError(f"CSV must have at least two columns: {csv_path}")
    if "true_label" in df.columns:
        true_series = df["true_label"]
    # prediction: prefer mc_pred_mode, else prob columns, else 2nd col
    if "mc_pred_mode" in df.columns:
        pred_vals = df["mc_pred_mode"]

    # normalize to ints
    y_true = pd.to_numeric(true_series, errors="coerce").fillna(-1).astype(int).values
    y_pred = pd.to_numeric(pred_vals, errors="coerce").fillna(-1).astype(int).values
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))   # <-- added this line
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    fpr = compute_macro_fpr(y_true, y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if "mc_pred_entropy" in df.columns:
        entropy = df["mc_pred_entropy"].values
    else:
        entropy = df.iloc[:, 2].values

    metrics = dict(
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1,
        fpr_macro=fpr,
        n_samples=len(y_true)
    )
    return metrics, cm, y_true, y_pred, entropy, pred_probs

def extract_reg_type_and_hyperparam(folder_name):
    """
    Returns (reg_type, value) where reg_type is "Ising", "Dropconnect", or "Dropout" depending
    on which hyperparameter is nonzero (priority: dc > bdo > do), or (None, None) if all zero/missing.
    """
    # Ising (dc)
    m = re.search(r"dc([0-9.]+)", folder_name)
    if m and float(m.group(1)) != 0.0:
        return "Ising", float(m.group(1))
    # Dropconnect (bdo)
    m = re.search(r"bdo[_\-]?([0-9.]+)", folder_name)
    if m and float(m.group(1)) != 0.0:
        return "Dropconnect", float(m.group(1))
    # Dropout (do)
    m = re.search(r"do([0-9.]+)", folder_name)
    if m and float(m.group(1)) != 0.0:
        return "Dropout", float(m.group(1))
    return None, None

# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if np.any(mask):
            acc = np.mean(y_true[mask] == (y_prob[mask] >= 0.5))
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask)
    return ece / len(y_prob)

#%%

DATASETS     = ["mnist", "fashionmnist", "cifar10", "cifar100"]
ISING_TYPES  = ["LM_saliency_scores", "no_saliency_scores"]
VAL_SPLITS   = [0.895, 0.8, 0.6, 0.1]
val_split_folders = [f"val{v:.3f}".rstrip("0").rstrip(".") for v in VAL_SPLITS]

# Traverse all folders, group by config, collect all run CSVs per config
all_results = []
all_entropies_by_case = {}
all_calibs_by_case = {}

# Each config folder (contains run CSVs)
for config_folder in [f for f in ckpt_root.iterdir() if f.is_dir() and f.name.startswith("checkpoint")]:
    for c1 in DATASETS:
        l1_fold = config_folder / c1
        if not l1_fold.is_dir():
            continue
        for c2 in ISING_TYPES:
            l2_fold = l1_fold / c2
            if not l2_fold.is_dir():
                continue
            for v in val_split_folders:
                l3_fold = l2_fold / v
                if not l3_fold.is_dir():
                    continue
                # l3_fold is now the config folder for this combination
                method, hyperparam = extract_reg_type_and_hyperparam(str(config_folder))
                run_csvs = [f for f in l3_fold.glob("*.csv") if "mc" in f.name]
                if not run_csvs:
                    continue
                # For each run, compute metrics
                conf_matrices = {} 

                for run_csv in run_csvs:
                    try:
                        metrics, cm, y_true, y_pred, entropy, pred_probs = process_pred_csv(run_csv)
                        if pred_probs.ndim == 2:
                            pred_class_probs = np.max(pred_probs, axis=1)  # confidence per sample
                            pred_class_label = np.argmax(pred_probs, axis=1)

                        # parse run_idx from file name if possible
                        m_run = re.search(r"run[_\-]?(\d+)", run_csv.name)
                        run_idx = int(m_run.group(1)) if m_run else None
                        row = dict(dataset=c1, method=method, hyperparam=hyperparam, val_split=v, run_idx=run_idx, file=str(run_csv))
                        row.update(metrics)
                        all_results.append(row)

                        # Save confusion matrix for this config
                        key = (c1, method, hyperparam, v)
                        conf_matrices.setdefault(key, []).append(cm)

                        # Append to the entropy dictionary
                        mask_correct = y_true == y_pred
                        mask_incorrect = y_true != y_pred
                        for case, mask in zip(["correct", "incorrect"], [mask_correct, mask_incorrect]):
                            all_entropies_by_case \
                                .setdefault(key, {}) \
                                .setdefault(method, {}) \
                                .setdefault(case, []).append(entropy[mask])
                            all_calibs_by_case \
                                .setdefault(key, {}) \
                                .setdefault(method, {}) \
                                .setdefault(case, []).append(pred_class_probs[mask])

                    except Exception as e:
                        print(f"Failed on {run_csv}: {e}")

                # ---- After the loop, compute mean/std confusion matrix for each config ----
                mean_std_confmats = {}  # (config) -> (mean, std) confusion matrices

                for key, mats in conf_matrices.items():
                    mats_arr = np.stack(mats, axis=0)  # shape: (n_runs, n_classes, n_classes)
                    mean_mat = mats_arr.mean(axis=0)
                    std_mat = mats_arr.std(axis=0)
                    mean_std_confmats[key] = (mean_mat, std_mat)

                # Example: print or save the confusion matrices for each config
                for key, (mean_mat, std_mat) in mean_std_confmats.items():
                    print(f"\nConfig: {key}")
                    print("Mean confusion matrix:\n", np.round(mean_mat, 2))
                    print("Std dev confusion matrix:\n", np.round(std_mat, 2))

                n_samples = row['n_samples']
                for key, (mean_mat, std_mat) in mean_std_confmats.items():
                    dataset, method, hyperparam, val_split = key
                    # vmax is the total number of samples divided by number of classes, assuming uniform
                    n_classes = mean_mat.shape[0]
                    vmax = n_samples / n_classes if n_classes > 1 else n_samples

                    figsize = (20, 16) if dataset == "cifar100" else (20, 16)
                    plt.figure(figsize=figsize)
                    plt.imshow(mean_mat, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)
                    plt.title(f"Mean Confusion Matrix\n{dataset}, {method}, hp={hyperparam}, val={val_split}, n={n_samples}")
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')

                    # Optional: annotate cells for small matrices
                    if n_classes <= 20:
                        for i in range(n_classes):
                            for j in range(n_classes):
                                plt.text(j, i, f"{mean_mat[i, j]:.1f}\n±{std_mat[i, j]:.1f}",
                                        ha='center', va='center', color='black', fontsize=8)
                    plt.colorbar()
                    plt.tight_layout()
                    # save
                    save_dir = os.path.join(out_dir, "confusion_matrices")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"confmat_{dataset}_{method}_hp{hyperparam}_{val_split}.png")
                    plt.savefig(save_path, dpi=120)
                    plt.close()
                    print(f"Saved: {save_path}")

methods = ["Dropconnect", "Dropout", "Ising"]
cases = ["correct", "incorrect"]

# Prepare mapping: (dataset, hyperparam, val_split) -> method -> {case: [arrays]}
entropy_plot_dict = defaultdict(lambda: {m: {'correct':[], 'incorrect':[]} for m in methods})
calib_plot_dict = defaultdict(lambda: {m: {'correct':[], 'incorrect':[]} for m in methods})

# ---- AGGREGATION LOOP - Entropy----
for key, method_dict in all_entropies_by_case.items():
    dataset, method, hyperparam, val_split = key
    base_key = (dataset, hyperparam, val_split)
    for case in cases:
        entropy_plot_dict[base_key][method][case].extend(method_dict.get(method, {}).get(case, []))  

# ---- AGGREGATION LOOP - Calibration----
for key, method_dict in all_calibs_by_case.items():
    dataset, method, hyperparam, val_split = key
    base_key = (dataset, hyperparam, val_split)
    for case in cases: 
        calib_plot_dict[base_key][method][case].extend(method_dict.get(method, {}).get(case, []))  

# ---- PLOTTING LOOP ----
for base_key, method_case_dict in entropy_plot_dict.items():
    dataset, hyperparam, val_split = base_key
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
    for row, method in enumerate(methods):
        for col, case in enumerate(cases):
            entropies_list = method_case_dict[method][case]
            entropies = np.concatenate(entropies_list) if entropies_list else np.array([])
            ax = axes[row, col]
            if len(entropies) > 0:
                ax.hist(entropies, bins=30, color='tab:blue' if case == "correct" else 'tab:orange', alpha=0.7, edgecolor='k')
            ax.set_title(f"{method} - {'Correct' if case == 'correct' else 'Incorrect'}")
            if row == 2:
                ax.set_xlabel("Entropy")
            if col == 0:
                ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Entropy Distribution (Correct vs Incorrect)\n{dataset}, hp={hyperparam}, val={val_split}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_dir = os.path.join(out_dir, "entropy_sbs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"entropy_sbs_{dataset}_hp{hyperparam}_{val_split}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")

# ---- CALIBRATION CURVES (Single Plot with Legends) ----
for base_key, method_vals in calib_plot_dict.items():
    dataset, hyperparam, val_split = base_key
    plt.figure(figsize=(8, 6))

    for method in methods:
        # Flatten `correct` and `incorrect` lists, ensuring dimensions are handled properly
        probs_correct = np.concatenate(method_vals[method]['correct']) if method_vals[method]['correct'] else np.array([])
        probs_incorrect = np.concatenate(method_vals[method]['incorrect']) if method_vals[method]['incorrect'] else np.array([])

        # Combine all samples for probabilities
        probs = np.concatenate([probs_correct, probs_incorrect])

        # Dynamically create `is_correct` array based on the number of samples in each group
        is_correct = np.concatenate([
            np.ones(probs_correct.shape[0]),  # 1 for all correct samples
            np.zeros(probs_incorrect.shape[0])  # 0 for all incorrect samples
        ]) if probs.size > 0 else np.array([])

        if len(probs) > 0:
            prob_true, prob_pred = calibration_curve(is_correct, probs, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=f"{method}")

    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')

    plt.title(f"Calibration Curves\n{dataset}, hp={hyperparam}, val={val_split}")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(out_dir, "calibration_combined")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"calibration_combined_{dataset}_hp{hyperparam}_{val_split}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")

# ---- RELIABILITY HISTOGRAMS (3,1 PLOT) ----
for base_key, method_case_dict in calib_plot_dict.items():
    dataset, hyperparam, val_split = base_key
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)

    for row, method in enumerate(methods):
        ax = axes[row]
        # Combine all confidences for the method (correct + incorrect)
        confidences = np.concatenate(method_case_dict[method]['correct'] + method_case_dict[method]['incorrect'])
        if len(confidences) > 0:
            counts, _ = np.histogram(confidences, bins=bins)
            centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(centers, counts / len(confidences), width=0.09, color='tab:blue', alpha=0.7, edgecolor='k')
        ax.set_title(f"{method}")
        ax.set_ylabel("Fraction of samples")
        ax.grid(True, alpha=0.3)
        if row == 2:
            ax.set_xlabel("Predicted confidence")
    fig.suptitle(f"Reliability Diagrams\n{dataset}, hp={hyperparam}, val={val_split}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_dir = os.path.join(out_dir, "reliability_sbs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"reliability_sbs_{dataset}_hp{hyperparam}_{val_split}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")

df = pd.DataFrame(all_results)
df.to_csv(out_dir / "per_run_metrics.csv", index=False)

# Group by (dataset, method, hyperparam, val_split), aggregate mean/std/count over runs
group_cols = ["dataset", "method", "hyperparam", "val_split"]
agg_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "fpr_macro"]
df_agg = df.groupby(group_cols)[agg_metrics].agg(['mean', 'std', 'count'])
df_agg.columns = ["_".join(c) for c in df_agg.columns]
df_agg = df_agg.reset_index()
df_agg.to_csv(out_dir / "aggregate_over_runs.csv", index=False)

# Pivot for comparison tables
for (dataset, val_split), group in df_agg.groupby(["dataset", "val_split"]):
    pivot = group.pivot(index="hyperparam", columns="method",
                        values=["accuracy_mean", "precision_macro_mean", "recall_macro_mean", "f1_macro_mean", "fpr_macro_mean"])
    pivot.to_csv(out_dir / f"comparison_{dataset}_{val_split}.csv")

print("Done. Results in", out_dir)