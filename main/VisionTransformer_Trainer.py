#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
import functools
import gc
import time
import random
from collections import defaultdict
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformer_layers.bbb_ViT import VisionTransformerWithBBB
from transformer_layers.bbb_linear import BBBLinear
from data_loader.dataloader_master import To3Channels, get_vit_dataloaders
from utils.early_stopping import EarlyStopping
from utils.learning_rate import adjust_learning_rate
from utils.metrics import metric, MAE, MSE, RMSE, MAPE, MSPE, LGLOSS, ACCRCY


def fast_compute_weight_dropout(final_layer, activations, targets, dropconnect_delta = 0.5, epsilon=1e-9, *, use_penalties=False, full_criterion=None):

    """
    Computes dropout probabilities for all weights in final linear layer
    by simulating zeroing each weight, all in parallel.
    """
    with torch.no_grad():
        W = final_layer.mean_weight           # (C, D)
        A = activations                               # (B, D)
        B, D = A.shape
        C = W.shape[0]

        logits = A @ W.T                              # (B, C)

        if use_penalties and full_criterion is not None:
            # one scalar total (includes penalties) – not recommended
            loss_orig = -full_criterion(logits, targets).repeat(len(targets)) # (B,)
        else:
            # per-example CE, *no penalties* (recommended)
            loss_orig = -F.cross_entropy(logits, targets, reduction='none')  # (B,)

        # Compute original class logit contributions
        contrib = torch.einsum('bd,cd->bcd', A, W)    # (B, C, D)

        # logits[b, c] - A[b, d] * W[c, d] for each d → perturbed logits
        logits_exp = logits.unsqueeze(1).unsqueeze(2)         # (B, 1, 1, C)
        logits_exp = logits_exp.expand(-1, C, D, -1)          # (B, C, D, C)
        eye   = torch.eye(C, device=W.device).view(1, C, 1, C) # eye : (1, C, 1, C) – broadcast helper

        delta = contrib.unsqueeze(-1) * eye                   # (B, C, D, C)
        logits_perturbed_full = logits_exp - delta            # (B, C, D, C)

        logits_flat  = logits_perturbed_full.reshape(B * C * D, C) # (B*C*D, C)

        targets_flat  = targets.view(B, 1, 1).expand(-1, C, D).reshape(-1)  # (B*C*D,)

        loss_dropped  = -F.cross_entropy(logits_flat, targets_flat, reduction='none')                  # (B*C*D,)
        loss_dropped  = loss_dropped.view(B, C, D)                          # (B, C, D) 
        
        # Loss difference per (d, b)
        delta_loss = loss_orig.view(B, 1, 1) - loss_dropped                 # (B, C, D)
        avg_delta  = delta_loss.mean(dim=0)                                 # (C, D)
        loss_diff = avg_delta                                              # (C, D)

        delta = np.log(dropconnect_delta/(1-dropconnect_delta))              # External Field
        dropout_prob = 1 - 1 / (1 + torch.exp(-2 * (0.5 * loss_diff) + delta))  # (C, D)

        return dropout_prob.detach()

def compute_diag_hessian_element(idx, grad1_flat, param, device):
    grad2 = torch.autograd.grad(
        grad1_flat[idx], param, retain_graph=True
    )[0]
    return grad2.view(-1)[idx].item()

class VisionTransformerTrainer:

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.models = self._build_model()
        self.models = [model.to(self.device) for model in self.models]      # Move each model to the appropriate device
        self.current_epoch = 'pilot'                                        # Initialize epoch tracker
        self._init_run_stats()          # Experiment results tracking
        
    def set_current_phase(self, phase):
        """
        Update the current epoch/phase for the model.
        """
        self.current_epoch = phase
        print(f"Current phase set to: {self.current_epoch}")

    def _init_run_stats(self):
        """Call once in __init__; prepares a blank container."""
        self._run_stats = {
            "dataset":        None,
            "train_samples":  None,
            "val_samples":    None,
            "test_samples":   None,
            "train_error":    None,   # final-epoch loss (1×M list)
            "val_error":      None,   # final-epoch loss (1×M list)
            "test_error":     None,   # final-epoch loss (1×M list)
            "train_acc":      None,   # 
            "val_acc":        None,   # 
            "test_acc":       None,   # 
            "train_err":      None,   # (100-acc)
            "val_err":        None,
            "test_err":       None,
            "num_parameters": None,
            "ising_expected_dropped":  None,
            "ising_dropped":  None,
            "total_potential":  None
        }

    def get_run_stats(self):
        """
        Call after `.train()`.  Returns a *copy* of the stats dict so
        external code can’t mutate the internal version by accident.
        """
        if not hasattr(self, "_run_stats"):
            raise RuntimeError("Run statistics not initialised; "
                               "make sure .train() has completed.")
        return self._run_stats.copy()

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Using GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device
   
    def _build_model(self):
        # Build multiple ViT models if specified in args
        models = []
        for _ in range(self.args.num_models):
            model = VisionTransformerWithBBB(
                img_size=self.args.img_size,
                patch_size=self.args.patch_size,
                num_classes=self.args.num_classes,  # Number of output classes
                embed_dim=self.args.embed_dim,
                num_heads=self.args.num_heads,
                depth=self.args.depth,
                dropout=self.args.dropout,
                dropconnect=self.args.dropconnect,
                device=self.device,
                epoch_tracker=self,  # Pass the instance for epoch tracking
            ).float()
    
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            models.append(model)
        return models

    def _get_data(self, flag):
        """
        Fetch the appropriate dataset and dataloader for the specified phase.
    
        Args:
            flag (str): Phase indicator ('train', 'val', 'test').
    
        Returns:
            tuple: (Dataset, DataLoader) for the specified phase.
        """
        args = self.args

        # Fetch all dataloaders using the pre-existing function
        dataloaders = get_vit_dataloaders(
            dataset_name=args.dataset,  # e.g., 'cifar10', 'cifar100', 'mnist', etc.
            data_dir=args.data_path,   # Path to the data directory
            batch_size=args.batch_size,
            val_split=args.val_split,  # Fraction of data for validation
            test_split=args.test_split,  # Fraction of data for testing
            image_size=args.img_size,  # Image resizing for ViT
            num_workers = args.num_workers
        )
    
        # Map flag to the correct dataset and dataloader
        if flag == 'train':
            data_set = dataloaders['train'].dataset
            data_loader = dataloaders['train']
        elif flag == 'val':
            data_set = dataloaders['val'].dataset
            data_loader = dataloaders['val']
        elif flag == 'test':
            data_set = dataloaders['test'].dataset
            data_loader = dataloaders['test']
        else:
            raise ValueError(f"Invalid flag: {flag}. Choose from 'train', 'val', or 'test'.")
    
        # Example: Log dataset stats for training phase
        if flag == 'train':
            self.train_pos = sum(1 for _, label in data_set if label == 1)  # Example for binary labels
            self.train_len = len(data_set)
            print(f"Training length: {self.train_len}")
    
        return data_set, data_loader

    def _get_data_ising(self, flag):
        """
        Fetch the appropriate dataset and dataloader for the specified phase.
    
        Args:
            flag (str): Phase indicator ('train', 'val', 'test').
    
        Returns:
            tuple: (Dataset, DataLoader) for the specified phase.
        """
        args = self.args

        # Fetch all dataloaders using the pre-existing function
        dataloaders = get_vit_dataloaders(
            dataset_name=args.dataset,  # e.g., 'cifar10', 'cifar100', 'mnist', etc.
            data_dir=args.data_path,   # Path to the data directory
            batch_size=1,
            val_split=args.val_split,  # Fraction of data for validation
            test_split=args.test_split,  # Fraction of data for testing
            image_size=args.img_size,  # Image resizing for ViT
            num_workers = args.num_workers
        )
    
        # Map flag to the correct dataset and dataloader
        if flag == 'train':
            data_set = dataloaders['train'].dataset
            data_loader = dataloaders['train']
        elif flag == 'val':
            data_set = dataloaders['val'].dataset
            data_loader = dataloaders['val']
        elif flag == 'test':
            data_set = dataloaders['test'].dataset
            data_loader = dataloaders['test']
        else:
            raise ValueError(f"Invalid flag: {flag}. Choose from 'train', 'val', or 'test'.")
    
        # Example: Log dataset stats for training phase
        if flag == 'train':
            self.train_pos = sum(1 for _, label in data_set if label == 1)  # Example for binary labels
            self.train_len = len(data_set)
    
        return data_set, data_loader

    def _calc_accuracy(self, data_loader, model):
        """
        Returns accuracy (%) of `model` on `data_loader`.
        """
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                out  = model(x)
                preds.append(out.argmax(dim=1).cpu().numpy())
                trues.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        return ACCRCY(preds, trues)          

    def _select_optimizer(self):
        # Create a list of optimizers for each model
        model_optim = [optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay = self.args.kl_pen) for model in self.models]

        return model_optim

    def _select_criterion(self):

        task_criterion = nn.CrossEntropyLoss()
    
        def criterion(predictions, targets):
            # Make sure targets are of type float for BCEWithLogitsLoss
            targets = targets.long()
            task_loss = task_criterion(predictions, targets)
            
            # Initialize similarity and orthogonal penalties
            similarity_penalty = 0.0
            orthogonal_penalty = 0.0
    
            # Multi-model case: add similarity penalty
            if self.args.num_models > 1:
                for i in range(len(self.models)):
                    for j in range(i + 1, len(self.models)):
                        for (name_i, param_i), (name_j, param_j) in zip(self.models[i].named_parameters(), self.models[j].named_parameters()):
                            # Compute pairwise similarity penalty between models
                            if self.args.sim_loss_type == 'root':
                                n = self.args.root  # Replace with any root
                                similarity_penalty += (torch.norm(param_i - param_j) + 1e-6) ** (1/n)
                            elif self.args.sim_loss_type == 'log':
                                similarity_penalty += torch.log(torch.norm(param_i - param_j) + 1e-6)  # Log of the Euclidean distance
                            elif self.args.sim_loss_type == 'orth':
                                param_i_vector = param_i.view(-1).unsqueeze(1)  # or param_i.flatten()
                                param_j_vector = param_j.view(-1).unsqueeze(1)  # or param_j.flatten()
                                identity = torch.eye(param_i_vector.size(0), device=param_i_vector.device)
                                similarity_penalty += torch.norm(torch.mm(param_i_vector, param_j_vector.T) - identity, p=2) ** 2   # Log of the Euclidean distance
    
            # Total loss = task loss - lambda_weight1 * similarity_penalty + lambda_weight2 * orthogonal_penalty
            lambda_weight1 = self.args.lambda_weight1
            lambda_weight2 = self.args.lambda_weight2
            total_loss = task_loss - lambda_weight1 * similarity_penalty + lambda_weight2 * orthogonal_penalty
            
            return total_loss

        return criterion  


    def vali(self, vali_loader, criterion, model):
        """
        Validation function for Vision Transformer.
    
        Args:
            vali_loader: DataLoader for validation data.
            criterion: Loss function for validation.
            model: The Vision Transformer model.
    
        Returns:
            Average validation loss.
        """
        model.eval()  # Set model to evaluation mode
        total_loss = []
    
        with torch.no_grad():  # Disable gradient computation
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                # Move data to the appropriate device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get predictions from the model
                pred = model(batch_x)  # Forward pass
                
                # Ensure predictions and targets are compatible with the criterion
                pred = pred.float()
                batch_y = batch_y.long()
    
                # Calculate loss
                loss = criterion(pred, batch_y)
                total_loss.append(loss.item())
    
        # Compute average loss
        avg_loss = np.mean(total_loss)
        model.train()  # Switch back to training mode
        return avg_loss

    def evaluate(self, save_pred=True, inverse=False, return_metrics=False):
        """
        Unified function to evaluate the model(s) on the test dataset.
        
        Args:
            save_pred (bool): Whether to save predictions and ground truths.
            inverse (bool): Whether to inverse the predictions if necessary.
            load_saved (bool): If True, assumes saved models are being evaluated (like eval).
            return_metrics (bool): If True, returns computed accuracy.
        
        Returns:
            float: Accuracy (if return_metrics=True).
        """
        args = self.args
    
        # Load dataset and dataloader
        data_set, data_loader = self._get_data(flag='test')
    
        accuracies_all = []  # Stores accuracy for each model
        all_preds = [[] for _ in range(self.args.num_models)]
        all_trues = [[] for _ in range(self.args.num_models)]
        
        # Evaluate each model
        for model_idx, model in enumerate(self.models):
            model.eval()
            all_pred_vals = []
            all_true_vals = []
    
            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
    
                    # Forward pass
                    pred = model(batch_x)
                    if inverse:
                        pred = self.inverse_transform(pred)
    
                    true = batch_y
    
                    # Convert predictions to class labels if necessary
                    pred_labels = torch.argmax(pred, dim=1) if pred.ndim > 1 else pred.round()
    
                    all_pred_vals.append(pred_labels.cpu().numpy())
                    all_true_vals.append(true.cpu().numpy())
    
            # Compute accuracy using ACCURACY function
            all_pred_vals = np.concatenate(all_pred_vals, axis=0)
            all_true_vals = np.concatenate(all_true_vals, axis=0)
            accuracy = ACCRCY(all_pred_vals, all_true_vals)
            accuracies_all.append(accuracy)
    
            print(f'Model {model_idx} - Accuracy: {accuracy:.2f}%')
    
            if save_pred:
                all_preds[model_idx] = all_pred_vals
                all_trues[model_idx] = all_true_vals
    
        # Save results
        folder_path = os.path.join(args.path, 'results')
        os.makedirs(folder_path, exist_ok=True)
    
        for model_idx in range(self.args.num_models):
            # Save accuracy to CSV
            metrics_df = pd.DataFrame({"Metrics": ["Accuracy"], "Values": [accuracies_all[model_idx]]})
            metrics_df.to_csv(os.path.join(folder_path, f'accuracy_model_{model_idx}.csv'), index=False)
    
            if save_pred:
                pd.DataFrame(all_preds[model_idx]).to_csv(os.path.join(folder_path, f'pred_model_{model_idx}.csv'), index=False)
                pd.DataFrame(all_trues[model_idx]).to_csv(os.path.join(folder_path, f'true_model_{model_idx}.csv'), index=False)
    
        if return_metrics:
            return accuracies_all

    
    def train(self):

        self.layer_inputs = {}  # Store inputs for each (model, layer) pair
        self._mask_history = defaultdict(list)   # layer_name ➜ [mask₁, …, maskₙ] (n≤100)
        avgd_masks = 0
    
        if self.args.ising_epochs > 0:
            def create_forward_hook(model, layer_name):
                def forward_hook(module, input, output):
                    # Store both the input and the layer name for better tracking
                    self.layer_inputs[(model, layer_name)] = input[0]
                        
                return forward_hook
            
            # Register hooks separately for each model instance
            for model in self.models:
                for name, layer in model.named_modules():
                    if isinstance(layer, BBBLinear):
                        layer.register_forward_hook(create_forward_hook(model, name))
            
        epsilon = 1e-9  # Small constant for numerical stability
        
        train_data_normal, train_loader_normal = self._get_data(flag = 'train') 
        vali_data_normal, vali_loader_normal = self._get_data(flag = 'val')
        test_data_normal, test_loader_normal = self._get_data(flag = 'test')

        if self.args.ising_batch == True:
            train_data_ising, train_loader_ising = self._get_data_ising(flag = 'train') 
            vali_data_ising, vali_loader_ising = self._get_data_ising(flag = 'val')
            test_data_ising, test_loader_ising = self._get_data_ising(flag = 'test') 

        train_steps = len(train_loader_normal)  

        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)
        ising_tag  = self.args.ising_type if self.args.ising_epochs > 0 else "none"
        train_size = len(train_data_normal)            # already computed
        ckpt_base  = os.path.join(
            path,
            f"checkpoint_{self.args.dataset}"
            f"_ising-{ising_tag}"
            f"_N{train_size}"
        )  
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, num_models=self.args.num_models, disable=self.args.disable_early_stopping)

        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
    
        # Track losses
        self.t_loss_tracker, self.v_loss_tracker, self.s_loss_tracker = [], [], []
    
        # Initialize for Ising computation
        self.ising_params = 0
        
        for epoch in range(self.args.train_epochs + self.args.ising_epochs + self.args.addtl_ft):
            # Identify training phase
            if epoch < self.args.train_epochs:
                phase = 'pilot'
            elif epoch < (self.args.train_epochs + self.args.ising_epochs):
                phase = 'ising'
            else:
                phase = 'fine-tuning'
            self.set_current_phase(phase)

            if ((phase == "ising") or (epoch == (self.args.train_epochs-1))) and (self.args.ising_batch==True):
                train_data, train_loader = train_data_ising, train_loader_ising
                vali_data, vali_loader = vali_data_ising, vali_loader_ising  
                test_data, test_loader = test_data_ising, test_loader_ising
            else:
                train_data, train_loader = train_data_normal, train_loader_normal
                vali_data, vali_loader = vali_data_normal, vali_loader_normal 
                test_data, test_loader = test_data_normal, test_loader_normal

            train_steps = len(train_loader) 

            # Initialize epoch variables
            time_now = time.time()
            iter_count = 0
            train_loss = [[] for _ in range(self.args.num_models)]
    
            # Set all models to training mode
            for model in self.models:
                model.train()

            epoch_time = time.time()

            for i, (batch_x,batch_y) in enumerate(train_loader):
                
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.to(self.device)
                iter_count += 1

                # Handle Ising phase-specific computations
                if phase == 'ising':

                    mask_list = []  # List to store masks for each layer in this model for the current batch
                    model = self.models[0] # Forward pass with the weight dropped
                    pred = model(batch_x)
                    final_layer = model.classification_head[-1]  # Assuming all models have the same final layer'  
                    final_layer_name = "classification_head.3"  # Ensure this matches the stored key

                    act = self.layer_inputs[(model, final_layer_name)]   # shape [B*T, D]

                    # B = batch_y.size(0)           # real batch size (1 during Ising phase)
                    # T = act.size(0) // B          # number of tokens per sample
                    # act = act.view(B, T, -1)      # [B, T, D]
                    # act = act.mean(dim=1)         # average over tokens  -> [B, D]

                    penultimate_act = act.detach()

                    if self.args.ising_type == "LM_saliency_scores":
                        saliency_scores = {}
                        deda = {}
                        grad = final_layer.mean_weight.grad  # Already computed during backprop
                        input_activations = self.layer_inputs[(self.models[0], final_layer_name)].mean(dim=0)
                        deda[final_layer_name] = 2*(grad / (input_activations + epsilon))**2  
                        saliency_scores[final_layer_name] = deda[final_layer_name].clone().detach()
                    

                    mask = fast_compute_weight_dropout(
                            final_layer   = final_layer,       # GPU weights
                            activations   = penultimate_act,   # GPU activations
                            targets       = batch_y,           # GPU labels
                            dropconnect_delta = self.args.dropconnect_delta,    # External Field Parameter
                            epsilon       = epsilon
                        )

                    mask_list.append(mask)                  # keep mask for later
                    final_layer.apply_custom_dropout_prob(mask)
                    
                    layer = None  # Initialize variable to store the first layer
                    captured_layers = []  # List to store all layers that were already captured
                    layer_counter = 0

                    attention_triplet = {'query':0, 'key':0, 'value':0}
                    # First loop to find layers with 'mean_weight'
                    for name, param in reversed(list(model.named_parameters())):                       
                        # Now use this list in your condition
                        if 'mean_weight' in name:
                            if layer is None:  # Check if the first layer has not been found yet
                                layer_path = name.replace('.mean_weight', '')  # Capture the first layer
                                layer_names = layer_path.split('.')  # Split the path into parts
                                layer_name = layer_path
                                layer = self.models[0]
                                
                                for ln in layer_names: # Navigate through the attributes
                                    if isinstance(layer, (nn.ModuleList, nn.Sequential)): # Check for nn.ModuleList or nn.Sequential to handle indexed layers
                                        if int(ln) < len(layer):
                                            layer = layer[int(ln)]  # Correctly handle indexing for layers like decode_layers[1]
                                    else: # Safeguard for getattr and debug info                             
                                        if hasattr(layer, ln):
                                            layer = getattr(layer, ln)
                                    
                                captured_layers.append(layer_path)  
                                
                            else:
                                next_layer_path = name.replace('.mean_weight', '')  # Capture the consecutive layer    
                                next_layer_names = next_layer_path.split('.')  # Split the path into parts
                                next_layer_name = next_layer_path
                                next_layer = self.models[0]

                                for ln in next_layer_names:
                                    if isinstance(next_layer, (nn.ModuleList, nn.Sequential)):
                                        if int(ln) < len(next_layer):
                                            next_layer = next_layer[int(ln)]  # Correctly handle indexing for layers like decode_layers[1]
                                    else:
                                        # Safeguard for getattr and debug info
                                        if hasattr(next_layer, ln):
                                            next_layer = getattr(next_layer, ln)            
                                
                                if ('key' in name):
                                    mask = mask_list[layer_counter-1]
                                elif('query' in name):
                                    mask = mask_list[layer_counter-2]
                                else:
                                    mask = mask_list[layer_counter]
                                
                                if 'query' in layer_names:
                                    
                                    mask_V = mask_list[layer_counter-2]
                                    mask_K = mask_list[layer_counter-1]
                                    mask_Q = mask_list[layer_counter]
                                    
                                    L_minus_1_connec =  ((2 * mask_V.t() - 1) * (attention_triplet['value']**2).t() + (2 * mask_K.t() - 1) * (attention_triplet['key']**2).t() + (2 * mask_Q.t() - 1) * (attention_triplet['query']**2).t())/(torch.sum((attention_triplet['value']**2).t(), dim=0, keepdim=True) + torch.sum((attention_triplet['key']**2).t(), dim=0, keepdim=True)+torch.sum((attention_triplet['query']**2).t(), dim=0, keepdim=True) + epsilon)
                                    
                                else:
                                    L_minus_1_connec =  (2 * mask.t() - 1) * (layer.mean_weight**2).t() / (torch.sum((layer.mean_weight**2).t(), dim=0, keepdim=True) + epsilon)

                                num_rows = next_layer.mean_weight.t().shape[0] # Get the number of rows in the next layer's mean_weight

                                if self.args.ising_type == "diag_saliency_scores":
                                    saliency_score = saliency_scores[name]
                                elif self.args.ising_type == "LM_saliency_scores":
                                    curr_param = next_layer.mean_weight
                                    prev_param = layer.mean_weight
                                    input_activations = self.layer_inputs[(self.models[0], next_layer_name)]
                
                                    # Collapse batch and sequence/patch dimensions to get per-unit input magnitude
                                    if input_activations.dim() == 3:
                                        # Shape: [batch_size, num_patches, hidden_dim] → mean over 0 and 1 → shape [hidden_dim]
                                        input_activations = input_activations.mean(dim=(0, 1))
                                    elif input_activations.dim() == 2:
                                        # Shape: [batch_size, hidden_dim] → mean over batch
                                        input_activations = input_activations.mean(dim=0)
                                    else:
                                        raise ValueError(f"Unexpected input_activations shape: {input_activations.shape}")
                                    
                                    # Make sure it's on the same device as the model parameters
                                    input_activations = input_activations.to(curr_param.device)
                                    deda[next_layer_name] = (curr_param.grad / (input_activations + 1e-8))**2  # Compute deda for the current layer
                                    # Propagate deda by weighting with the square of next layer's weights
                                    deda[next_layer_name] = torch.matmul(deda[next_layer_name].T, torch.matmul(prev_param.T ** 2, deda[layer_name])).T 
                                    saliency_score = 0.5 * deda[next_layer_name].clone().detach() * (curr_param ** 2)

                                elif self.args.ising_type == "no_saliency_scores":
                                    L1_mat = torch.sum(L_minus_1_connec, dim=1).unsqueeze(0).repeat(num_rows, 1)
                                    saliency_score = torch.zeros_like(L1_mat.t())
                                    
                                a_i_tensor = torch.sum(L_minus_1_connec, dim=1).unsqueeze(0).repeat(num_rows, 1) + 0.5*saliency_score.t()
                                L_minus_1_dropout_probs = 1 - 1 / (1 + torch.exp(-2 * a_i_tensor + np.log(self.args.dropconnect_delta/(1-self.args.dropconnect_delta))))
                                L_minus_1_dropout_probs = L_minus_1_dropout_probs.detach()
                                mask_list.append(L_minus_1_dropout_probs.t())  # Save masks for the current layer in the current batch
                                next_layer.apply_custom_dropout_prob(L_minus_1_dropout_probs.t())

                                if('value' in name):
                                    captured_layers.append(next_layer_path)
                                    attention_triplet['value'] = next_layer.mean_weight
                                elif('key' in name):
                                    captured_layers.append(next_layer_path)
                                    attention_triplet['key'] = next_layer.mean_weight
                                elif('query' in name):
                                    attention_triplet['query'] = next_layer.mean_weight
                                    layer_path = next_layer_path  # Update 'layer_name' to the latest found layer
                                    captured_layers.append(layer_path)  
                                    layer = next_layer 
                                    layer_name = next_layer_name
                                    layer_names = layer_path.split('.')  # Split the path into parts
                                else:
                                    layer_path = next_layer_path  # Update 'layer_name' to the latest found layer
                                    captured_layers.append(layer_path)  
                                    layer = next_layer
                                    layer_name = next_layer_name
                                    layer_names = layer_path.split('.')  # Split the path into parts
                                
                                layer_counter += 1

                    for lname, lmask in zip(captured_layers, mask_list):
                        hist = self._mask_history[lname]
                        hist.append(lmask.detach().cpu())      # store on CPU
                        if len(hist) > 100:                    # clamp length
                            hist.pop(0)            
                
                # Handle Ising phase-specific computations
                if phase == 'fine-tuning' and avgd_masks == 0:
                    avgd_masks = 1

                    for layer_name, masks in self._mask_history.items():
                        if not masks:                 # skip layers with no history
                            continue
                        avg_mask = torch.stack(masks).mean(0)

                        # Threshold to binary mask at 0.5
                        binary_mask = (avg_mask > self.args.drop_thresh).float()

                        for model in self.models:
                            mod = model
                            for part in layer_name.split('.'):      # navigate to layer
                                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)

                            mod.register_buffer("avg_dropout_mask", avg_mask.to(mod.mean_weight.device))
                            mod.apply_custom_dropout_prob(mod.avg_dropout_mask)
                    
                    # ---------- Ising hard-drop summary based on final masks ----------   NEW
                    expec_dropped, hard_dropped, total_masked = 0, 0, 0
                    if self.args.ising_epochs > 0:
                        for model in self.models:
                            for name, module in model.named_modules():
                                if hasattr(module, "avg_dropout_mask"):
                                    mask = module.avg_dropout_mask
                                    expec_dropped += (mask > 0.5).sum().item()
                                    hard_dropped += (mask > self.args.drop_thresh).sum().item()
                                    total_masked += mask.numel()

                        print(f"Ising expected dropped params: {expec_dropped} "
                            f"({100 * expec_dropped / total_masked:.2f}% of {total_masked})")

                        print(f"Ising hard-threshold dropped params: {hard_dropped} "
                            f"({100 * hard_dropped / total_masked:.2f}% of {total_masked})")

                        num_weights = sum(p.numel() for p in self.models[0].parameters())
                        print(f"Total model parameters: {num_weights}")

                    # Run stats
                    self.ising_expec_params = expec_dropped
                    self.ising_params = hard_dropped
                    self.total_maskable = total_masked

                # Training logic (core loop)
                for optimizer in model_optim: # Zero the gradients for each model's optimizer
                    optimizer.zero_grad()
                    
                shuffled_indices = list(range(len(self.models))) 
                random.shuffle(shuffled_indices)

                for model_idx in shuffled_indices: # Loop over models randomly and process batches
                    model = self.models[model_idx]
                    pred = model(batch_x)  
    
                    # Compute loss for each model
                    loss = criterion(pred, batch_y)
                    train_loss[model_idx].append(loss.item())

                    if ((epoch == (self.args.train_epochs-1)) and (i == (train_steps-1))) or ((self.current_epoch == 'ising') and (i == (train_steps-1))):
                        
                        if self.args.ising_type == "diag_saliency_scores":
                            saliency_scores = {}  # Dictionary to store saliency scores for each parameter
                            # My Hessian implementation
                            for name, param in model.named_parameters():
                                if 'mean_weight' in name:
                                    # Zero gradients before backward pass
                                    for optimizer in model_optim:
                                        optimizer.zero_grad()
                                    
                                    # Compute first derivative
                                    grad1 = torch.autograd.grad(loss, param, create_graph=True)[0]
        
                                    # Initialize diagonal Hessian
                                    diag_hessian = torch.zeros_like(param)
                            
                                    # Compute diagonal of Hessian
                                    for idx in range(param.numel()):
                                        # Compute the second derivative
                                        grad2 = torch.autograd.grad(grad1.view(-1)[idx], param, retain_graph=True)[0]
                                        diag_hessian.view(-1)[idx] = grad2.view(-1)[idx]
                                    
                                    # Store saliency scores
                                    saliency_scores[name] = (0.5 * diag_hessian * (param ** 2)).detach()

                    for optimizer in model_optim:
                        optimizer.zero_grad() # zero the gradients

                    loss.backward()  # Backpropagation
                    model_optim[model_idx].step()  # Update model

                if (i + 1) % 100 == 0:
                    avg_losses = [np.mean(losses) for losses in train_loss]
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss for each model: {avg_losses}")
                    speed = (time.time() - time_now) / iter_count
                    print('\tspeed: {:.4f}s/iter'.format(speed))
                    iter_count = 0
                    time_now = time.time()    

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))

            # Validation and early stopping          
            train_loss_avg = [np.mean(losses) for losses in train_loss]
            vali_loss_avg = [self.vali(vali_loader, criterion, model) for model in self.models]  # Pass each model
            test_loss_avg = [self.vali(test_loader, criterion, model) for model in self.models]  # Pass each model

            self.t_loss_tracker.append(train_loss_avg)
            self.v_loss_tracker.append(vali_loss_avg)
            self.s_loss_tracker.append(test_loss_avg)
            
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss_avg}, Vali Loss: {vali_loss_avg}, Test Loss: {test_loss_avg}")
            early_stopping(vali_loss_avg, self.models, ckpt_base, phase)  # Early stopping on model[0] as reference
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust learning rates for each optimizer
            for optimizer in model_optim:
                adjust_learning_rate(optimizer, epoch + 1, self.args)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Save all M models separately
        for idx, model in enumerate(self.models):
            best_model_path = f"{ckpt_base}_model_{idx}.pth"
            model.load_state_dict(torch.load(best_model_path))
            state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            torch.save(state_dict, best_model_path)  # Save each model's state dict     
        
        # ------------------------------------------------------------------
        #                 FINALISE   R U N   S U M M A R Y
        # ------------------------------------------------------------------
        # Dataset & split sizes -------------------------------------------
        self._run_stats["dataset"]       = self.args.dataset
        self._run_stats["train_samples"] = len(train_data_normal)
        self._run_stats["val_samples"]   = len(vali_data_normal)
        self._run_stats["test_samples"]  = len(test_data_normal)

        # Final-epoch errors (average CE losses already tracked) ----------
        # Each entry is a list of length = num_models
        self._run_stats["train_error"] = self.t_loss_tracker[-1]
        self._run_stats["val_error"]   = self.v_loss_tracker[-1]
        self._run_stats["test_error"]  = self.s_loss_tracker[-1]

        # ----------  accuracy on last epoch  ----------
        train_acc = [self._calc_accuracy(train_loader_normal, m) for m in self.models]
        val_acc   = [self._calc_accuracy(vali_loader_normal,  m) for m in self.models]
        test_acc  = [self._calc_accuracy(test_loader_normal,   m) for m in self.models]

        self._run_stats["train_acc"] = train_acc
        self._run_stats["val_acc"]   = val_acc
        self._run_stats["test_acc"]  = test_acc

        self._run_stats["train_err"] = [100 - a for a in train_acc]
        self._run_stats["val_err"]   = [100 - a for a in val_acc]
        self._run_stats["test_err"]  = [100 - a for a in test_acc]

        # Parameter counts & Ising statistics -----------------------------
        # `self.total_params` is set during the last Ising batch;
        # fall back to a simple count if Ising wasn’t run.
        total_params = getattr(self, "total_params",
                               sum(p.numel() for p in self.models[0].parameters()))
        self._run_stats["num_parameters"] = total_params
        self._run_stats["ising_expected_dropped"]  = getattr(self, "ising_expec_params", 0)
        self._run_stats["ising_dropped"]  = getattr(self, "ising_params", 0)
        self._run_stats["total_potential"]  = getattr(self, "total_maskable", 0)


        return self.models

