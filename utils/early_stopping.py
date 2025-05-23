import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, num_models=1):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.num_models = num_models
        self.counter = 0
        self.early_stop = False
        self.best_scores = [None] * num_models
        self.val_loss_mins = [np.inf] * num_models
        self.phase = None

    def __call__(self, val_loss, models, path, phase=None):
        if phase and phase != getattr(self, "phase", None):
            print(f"🔁 Phase changed from {getattr(self, 'phase', 'None')} → {phase}")
            if phase == "fine-tuning":
                print("🔄 Resetting early stopping history for fine-tuning phase...")
                self.best_scores = [None] * self.num_models
                self.val_loss_mins = [np.inf] * self.num_models
                self.counter = 0
                self.early_stop = False
            self.phase = phase
        # Ensure val_loss is a list or array
        if isinstance(val_loss, (int, float)):
            val_loss = [val_loss]  # Convert scalar to list

        # Track if any model has improved
        any_model_improved = False
        
        for model_idx in range(self.num_models):
            score = -val_loss[model_idx]
            old_val_loss = self.val_loss_mins[model_idx]  # Fetch previous best loss
            if self.best_scores[model_idx] is None:
                self.best_scores[model_idx] = score
                self.val_loss_mins[model_idx] = val_loss[model_idx]
                self.save_checkpoint(val_loss[model_idx], models[model_idx], path, model_idx, old_val_loss)
                any_model_improved = True
            elif phase == "ising":
                # Always save during ising phase
                if self.verbose:
                    print(f"Saving model {model_idx} during 'ising' phase regardless of loss.")
                self.save_checkpoint(val_loss[model_idx], models[model_idx], path, model_idx, old_val_loss)

            elif score >= self.best_scores[model_idx] + self.delta:
                self.best_scores[model_idx] = score
                self.val_loss_mins[model_idx] = val_loss[model_idx]
                self.save_checkpoint(val_loss[model_idx], models[model_idx], path, model_idx, old_val_loss)
                any_model_improved = True
            

        # Check if early stopping should be triggered
        if any_model_improved:
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, path, model_idx, old_val_loss):
        if self.verbose:
            print(f'Model {model_idx}: Validation loss decreased ({old_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f"{path}_model_{model_idx}.pth")
        self.val_loss_mins[model_idx] = val_loss
