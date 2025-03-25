#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import random
from torch.nn import DataParallel
from joblib import Parallel, delayed
from transformer_layers.bbb_ViT import VisionTransformerWithBBB
from transformer_layers.bbb_linear import BBBLinear
from data_loader.dataloader_master import To3Channels, get_vit_dataloaders
from utils.early_stopping import EarlyStopping
from utils.learning_rate import adjust_learning_rate
from utils.metrics import metric, MAE, MSE, RMSE, MAPE, MSPE, LGLOSS, ACCRCY


def compute_weight_dropout(nr, nc, model, batch_x, batch_y, loss_NoDrop_item, criterion, epsilon, device):

    model_p = copy.deepcopy(model).to(device).eval()
    final_layer = model_p.classification_head[-1]

    # Drop the weight
    final_layer.mean_weight[nr, nc].data.zero_()

    with torch.no_grad():
        output_dropped = model_p(batch_x.to(device))
        loss_dropped = -criterion(output_dropped, batch_y.to(device))
        loss_difference = 0.5*(loss_NoDrop_item - loss_dropped.item()) + epsilon
        dropout_prob = 1 - 1 / (1 + torch.exp(-2 * loss_difference))

    return dropout_prob.item()


class VisionTransformerTrainer:

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.models = self._build_model()

        # Move each model to the appropriate device
        self.models = [model.to(self.device) for model in self.models]
        # Initialize epoch tracker
        self.current_epoch = 'pilot'
        self.epoch_mask_list = []
        
    def set_current_phase(self, phase):
        """
        Update the current epoch/phase for the model.
        """
        self.current_epoch = phase
        print(f"Current phase set to: {self.current_epoch}")

    def _add_epoch_mask(self, mask):
        """
        Append a mask to the epoch mask list for the current epoch.
        """
        self.epoch_mask_list.append(mask)
        print(f"Mask added for epoch '{self.current_epoch}': {mask}")

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
            image_size=args.img_size  # Image resizing for ViT
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

    def find_out_projection(layer):
        """
        Recursively searches for the 'out_projection' layer within a given module.
        Useful for locating output projection layers in Vision Transformers or other nested models.
        
        Args:
            layer (torch.nn.Module): The module to search through.
    
        Returns:
            torch.nn.Module or None: The 'out_projection' layer if found; otherwise, None.
        """
        # If the current layer has an out_projection attribute, return it
        if hasattr(layer, 'out_projection'):
            return layer.out_projection
    
        # If it's a container (e.g., Sequential, ModuleList), search its children recursively
        for sublayer in layer.children():
            result = find_out_projection(sublayer)
            if result is not None:
                return result
    
        # Return None if 'out_projection' is not found in this layer or its children
        return None

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

    def evaluate(self, save_pred=True, inverse=False, load_saved=False, return_metrics=False):
        """
        Unified function to evaluate the model(s) on the test dataset.
        
        Args:
            save_pred: Whether to save predictions and ground truths.
            inverse: Whether to inverse the predictions if necessary.
            load_saved: If True, assumes saved models are being evaluated (like eval).
            return_metrics: If True, returns the computed metrics (like eval).
        """
        args = self.args
    
        # Load dataset and dataloader
        if load_saved:
            # Explicit dataset setup for saved models
            data_set = Dataset_gen(
                root_path=args.path,
                data_path=args.data_path,
                flag='test',
                size=[args.in_len, args.out_len],
                data_split=args.data_split,
                scale=args.scale,
                scale_statistic=args.scale_statistic,
            )
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
        else:
            # Use the existing `_get_data` method
            data_set, data_loader = self._get_data(flag='test')
    
        metrics_all = [[] for _ in range(self.args.num_models)]
        all_preds = [[] for _ in range(self.args.num_models)]
        all_trues = [[] for _ in range(self.args.num_models)]
        instance_num = 0
    
        # Evaluate each model
        for model_idx, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
    
                    # Forward pass
                    pred = model(batch_x)
                    if inverse:
                        pred = self.inverse_transform(pred)
    
                    true = batch_y
                    batch_size = pred.size(0)
                    instance_num += batch_size
    
                    # Calculate metrics
                    batch_metric = np.array(
                        metric(pred.cpu().numpy(), true.cpu().numpy())
                    ) * batch_size
                    metrics_all[model_idx].append(batch_metric)
    
                    if save_pred:
                        all_preds[model_idx].append(pred.cpu().numpy())
                        all_trues[model_idx].append(true.cpu().numpy())
    
        # Save results
        folder_path = os.path.join(args.path, 'results')
        os.makedirs(folder_path, exist_ok=True)
    
        for model_idx in range(self.args.num_models):
            metrics_all[model_idx] = np.stack(metrics_all[model_idx], axis=0)
            metrics_mean = metrics_all[model_idx].sum(axis=0) / instance_num
    
            mae, mse, rmse, mape, mspe, lgls = metrics_mean
            print(f'Model {model_idx} - CBE: {lgls}')
    
            # Save metrics
            metrics_df = pd.DataFrame(
                {"Metrics": ["MAE", "MSE", "RMSE", "MAPE", "MSPE", "CBE"],
                 "Values": [mae, mse, rmse, mape, mspe, lgls]}
            )
            metrics_df.to_csv(os.path.join(folder_path, f'metrics_model_{model_idx}.csv'), index=False)
    
            if save_pred:
                preds = np.concatenate(all_preds[model_idx], axis=0)
                trues = np.concatenate(all_trues[model_idx], axis=0)
    
                pd.DataFrame(preds).to_csv(os.path.join(folder_path, f'pred_model_{model_idx}.csv'), index=False)
                pd.DataFrame(trues).to_csv(os.path.join(folder_path, f'true_model_{model_idx}.csv'), index=False)
    
        # Optionally return metrics for programmatic use
        if return_metrics:
            return [mae, mse, rmse, mape, mspe, lgls]
    
    def train(self):

        self.layer_inputs = {}  # Store inputs for each (model, layer) pair
    
        if self.args.ising_type == "LM_saliency_scores":
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
        
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, num_models = self.args.num_models)
        
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

            # Initialize epoch variables
            time_now = time.time()
            iter_count = 0
            train_loss = [[] for _ in range(self.args.num_models)]
    
            # Set all models to training mode
            for model in self.models:
                model.train()

            epoch_time = time.time()
            batch_masks = []  # List to store masks for each model in the current batch

            for i, (batch_x,batch_y) in enumerate(train_loader):

                batch_y = batch_y.to(self.device)
                iter_count += 1

                # Handle Ising phase-specific computations
                if phase == 'ising':
                    
                    mask_list = []  # List to store masks for each layer in this model for the current batch
                    original_loss = loss.item()  # Store the loss before modifying the weights
                    final_layer = self.models[0].classification_head[-1]  # Assuming all models have the same final layer'  
                    final_layer_name = "classification_head.3"  # Ensure this matches the stored key
                    weight_dropout_probs = [] # Calculate dropout probabilities for weights
                    loss_NoDrop = -criterion(pred, batch_y) # Taking negative to convert to log likelihood instead of negative log-likelihood 

                    if self.args.ising_type == "LM_saliency_scores":
                        saliency_scores = {}
                        deda = {}
                        grad = final_layer.mean_weight.grad  # Already computed during backprop
                        input_activations = self.layer_inputs[(self.models[0], final_layer_name)]
                        deda[final_layer_name] = 2*(grad / (input_activations + epsilon))**2  
                        saliency_scores[final_layer_name] = deda[final_layer_name].clone()
                    
                    for nr in range(final_layer.mean_weight.shape[0]):  # Loop through each output weight
                        for nc in range(final_layer.mean_weight.shape[1]):  # Loop through each input feature
                            original_weight = final_layer.mean_weight[nr, nc].item() # save original weight
                            final_layer.mean_weight[nr, nc].data = torch.tensor(0.0, device=final_layer.mean_weight.device) # Drop the weight temporarily
                            model = self.models[0] # Forward pass with the weight dropped
                            
                            output_dropped = model(batch_x)
                            
                            loss_dropped = -criterion(output_dropped, batch_y) # Compute cross-entropy loss under dropped weights
                            loss_difference = 0.5*(loss_NoDrop.item() - loss_dropped.item()) + epsilon # Calculate loss difference and compare with first model's loss
                            
                            a_i = loss_difference
                            a_i_tensor = torch.tensor(a_i) if isinstance(a_i, float) else a_i
                            dropout_prob = 1 - 1 / (1 + torch.exp(-2*a_i_tensor)) # Calculate dropout probability
                            weight_dropout_probs.append(dropout_prob)

                            final_layer.mean_weight[nr, nc].data = torch.tensor(original_weight, device=final_layer.mean_weight.device) # Restore the original weight
    
                    weight_dropout_probs = torch.tensor(weight_dropout_probs).to(final_layer.mean_weight.device) # Convert list to tensor for easier manipulation

                    mask = weight_dropout_probs.view(final_layer.mean_weight.shape).detach()
                    
                    mask_list.append(mask) # save masks for last layer in current batch
                    final_layer.apply_custom_dropout_prob(mask) # Pass the mask to the final layer for dropout
                    
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
                                    if input_activations.dim() == 3:
                                        input_activations = input_activations.sum(dim=1)
                                    deda[next_layer_name] = (curr_param.grad / (input_activations + 1e-8))**2  # Compute deda for the current layer
                                    # Propagate deda by weighting with the square of next layer's weights
                                    deda[next_layer_name] = torch.matmul(deda[next_layer_name].T, torch.matmul(prev_param.T ** 2, deda[layer_name])).T 
                                    saliency_score = 0.5 * deda[next_layer_name].clone() * (curr_param ** 2)

                                elif self.args.ising_type == "no_saliency_scores":
                                    L1_mat = torch.sum(L_minus_1_connec, dim=1).unsqueeze(0).repeat(num_rows, 1)
                                    saliency_score = torch.zeros_like(L1_mat.t())
                                    
                                a_i_tensor = torch.sum(L_minus_1_connec, dim=1).unsqueeze(0).repeat(num_rows, 1) + 0.5*saliency_score.t()
                                L_minus_1_dropout_probs = 1 - 1 / (1 + torch.exp(-2 * a_i_tensor))
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
                                
                    batch_masks.append(mask_list)
                    if (i == (train_steps-1)) and (epoch == (self.args.train_epochs+self.args.ising_epochs-1)):
                        total_ones = 0  # Counter for total number of ones across all tensors                        
                        for fmask in mask_list: # Loop through each tensor in the mask_list
                            final_mask = (fmask < 0.5).int() # Apply the threshold: elements < 0.5 become 0, the rest become 1
                            total_ones += torch.sum(final_mask).item() # Count the number of ones in the current tensor and add to the total
                        print(f'Ising dropped params:{total_ones}')
                        num_weights = sum(p.numel() for p in model.parameters())
                        print(f'Total num params:{num_weights}')
                        self.ising_params = total_ones

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
                                    saliency_scores[name] = 0.5 * diag_hessian * (param ** 2)

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

            # Store the masks for this epoch in the main epoch list
            self.epoch_mask_list.append(batch_masks)

            self.t_loss_tracker.append(train_loss_avg)
            self.v_loss_tracker.append(vali_loss_avg)
            self.s_loss_tracker.append(test_loss_avg)
            
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss_avg}, Vali Loss: {vali_loss_avg}, Test Loss: {test_loss_avg}")
            early_stopping(vali_loss_avg, self.models, path, phase)  # Early stopping on model[0] as reference
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust learning rates for each optimizer
            for optimizer in model_optim:
                adjust_learning_rate(optimizer, epoch + 1, self.args)

        # Save all M models separately
        for idx, model in enumerate(self.models):
            best_model_path = path + f'/checkpoint_S-BICF_model_{idx}.pth'
            model.load_state_dict(torch.load(best_model_path))
            state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            torch.save(state_dict, best_model_path)  # Save each model's state dict     
        
        
        return self.models

