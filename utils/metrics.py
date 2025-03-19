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