import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def LGLOSS(pred, true):
    
    # Ensure predictions are clipped to avoid log(0)
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    
    # Compute Log Loss
    loss = -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))
    
    return loss
    
def ACCRCY(pred, true):
    """
    Computes accuracy as the percentage of exactly matching values.
    
    Args:
        pred (numpy.ndarray): Predicted values.
        true (numpy.ndarray): Ground truth values.
    
    Returns:
        float: Accuracy as a percentage.
    """
    correct = np.sum(pred == true)  # Count exact matches
    total = true.size  # Total number of samples
    return (correct / total) * 100  # Return percentage accuracy

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    lgls = LGLOSS(pred, true)
    acc = ACCRCY(pred, true)  # Strict accuracy computation
    
    return mae, mse, rmse, mape, mspe, lgls, acc  # Include strict accuracy in the return values


def multiclass_calibration_metrics(probs, true, n_bins=15):
    """Compute ECE, MCE, and multiclass Brier score from class probabilities.

    ECE (expected calibration error) is the confidence-weighted average gap
    between empirical accuracy and mean confidence across bins. MCE is the
    largest such bin gap. The Brier score is the mean squared error between
    the probability vector and the one-hot encoded class label.
    """
    probs = np.asarray(probs, dtype=float)
    true = np.asarray(true, dtype=int)
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape (N, C); got {probs.shape}")
    if true.ndim != 1 or true.shape[0] != probs.shape[0]:
        raise ValueError("true must be a length-N integer vector matching probs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    row_sums = probs.sum(axis=1)
    if not np.all(np.isfinite(probs)) or not np.all(np.isfinite(row_sums)):
        raise ValueError("probs contains NaN or Inf")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probs must lie in [0, 1]")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(true)
    for bin_idx in range(n_bins):
        lo, hi = edges[bin_idx], edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (count / n) * gap
        mce = max(mce, gap)

    one_hot = np.eye(probs.shape[1], dtype=float)[true]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return {"ece": float(ece), "mce": float(mce), "brier": float(brier)}
