class Ising_Exp_transformer(Ising_Exp_Basic):
    def __init__(self, args):
        super(Ising_Exp_transformer, self).__init__(args)
        self.current_epoch = 'pilot'  # Track the current epoch
        # Initialize the list to store masks across epochs
        self.epoch_mask_list = []

    def set_current_phase(self, phase):
        self.current_epoch = phase
    
    def _build_model(self):        
        # Build M models
        models = []
        for _ in range(self.args.num_models):
            model = Transformer(
                self.args.data_dim, 
                self.args.in_len, 
                self.args.out_len,
                self.args.seg_len,
                self.args.win_size,
                self.args.factor,
                self.args.d_model, 
                self.args.d_ff,
                self.args.n_heads, 
                self.args.e_layers,
                self.args.dropout, 
                self.args.baseline,
                self.device,
                epoch_tracker=self  # Pass in the model instance itself to track epoch
            ).float()
            
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            models.append(model)
        return models

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
            
        data_set = Dataset_gen(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = args.scale,
            smooth = args.smooth
        )
        
        if flag == 'train':
            self.train_pos = np.sum(data_set.labels)
            self.train_len = len(data_set)

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        # Create a list of optimizers for each model
        model_optim = [optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay = self.args.kl_pen) for model in self.models]

        return model_optim
    
    def _select_criterion(self):
    
        # Calculate class weights once
        total_count = self.train_len
        class_weights = torch.tensor([
            total_count / (self.train_len - self.train_pos),  # weight for negative class
            total_count / self.train_pos  # weight for positive class
        ], dtype=torch.float)
    
        # Define the weighted BCEWithLogitsLoss using class_weights
        task_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1] / class_weights[0])
        # task_criterion = nn.BCEWithLogitsLoss()
    
        def criterion(predictions, targets):
            # Make sure targets are of type float for BCEWithLogitsLoss
            targets = targets.float()
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
                            elif self.args.sim_loss_type == 'orth_red':

                                cond1 = (("encoder.encode_blocks.0.encode_layers.0.router" in name_i) and ("encoder.encode_blocks.0.encode_layers.0.router" in name_j))
                                cond2 = (("decoder.decode_layers.0.self_attention.router" in name_i) and ("decoder.decode_layers.0.self_attention.router" in name_j))

                                if (cond1 or cond2):
                                    param_i_vector = param_i.view(-1).unsqueeze(1)  # or param_i.flatten()
                                    param_j_vector = param_j.view(-1).unsqueeze(1)  # or param_j.flatten()
                                    identity = torch.eye(param_i_vector.size(0), device=param_i_vector.device)
                                    similarity_penalty += torch.norm(torch.mm(param_i_vector, param_j_vector.T) - identity, p=2) ** 2   # Log of the Euclidean distance

                            # Apply orthogonalization penalty for the specific layer
                            if "decoder.decode_layers.0.self_attention.router" in name_i:
                                weight_i = param_i.view(param_i.size(1), param_i.size(2)) if param_i.dim() > 2 else param_i
                                identity = torch.eye(weight_i.size(0), device=weight_i.device)
                                # print('Weight Shape')
                                # print(weight_i.shape)
                                orthogonal_penalty += torch.norm(torch.mm(weight_i, weight_i.T) - identity, p=2) ** 2 


            else:
                # Compute orthogonal penalty
                for model in self.models:
                    for name_i, param_i in model.named_parameters():
                        # Apply orthogonalization penalty for the specific layer
                        if "decoder.decode_layers.0.self_attention.router" in name_i:
                            weight_i = param_i.view(param_i.size(1), param_i.size(2)) if param_i.dim() > 2 else param_i
                            identity = torch.eye(weight_i.size(0), device=weight_i.device)

                            orthogonal_penalty += torch.norm(torch.mm(weight_i, weight_i.T) - identity, p=2) ** 2   
    
            
            # Total loss = task loss - lambda_weight1 * similarity_penalty + lambda_weight2 * orthogonal_penalty
            lambda_weight1 = self.args.lambda_weight1
            lambda_weight2 = self.args.lambda_weight2
            total_loss = task_loss - lambda_weight1 * similarity_penalty + lambda_weight2 * orthogonal_penalty
            
            return total_loss
    
        return criterion  

    def find_out_projection(layer):
        # If the layer itself has an out_projection, return it
        if hasattr(layer, 'out_projection'):
            return layer.out_projection
        
        # If it's a container like Sequential or ModuleList, loop through its children
        for sublayer in layer.children():
            result = find_out_projection(sublayer)
            if result is not None:
                return result
        
        return None
    
    def vali(self, vali_data, vali_loader, criterion, model):
        model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, model)
                
                pred = pred.float().squeeze()
                true = true.long().squeeze()
                
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, num_models = self.args.num_models)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        self.t_loss_tracker = []
        self.v_loss_tracker = []
        self.s_loss_tracker = []

        self.ising_params = 0
        
        # for epoch in range(self.args.train_epochs + self.args.ising_epochs + 50):
        for epoch in range(self.args.train_epochs+self.args.ising_epochs+50):

            # Ising epochs identifier
            if epoch < self.args.train_epochs:
                phase = 'pilot'
            elif epoch < (self.args.train_epochs + self.args.ising_epochs):
                phase = 'ising'
            else:
                phase = 'fine-tuning'

            self.set_current_phase(phase)  # Update the current phase in Ising_Exp_transformer class
            
            # At the start of a new training epoch
            time_now = time.time()
            iter_count = 0
            train_loss = [[] for _ in range(self.args.num_models)]
            
            # Set all models to training mode
            for model in self.models:
                model.train()
                
            epoch_time = time.time()

            batch_masks = []  # List to store masks for each model in the current batch
            
            for i, (batch_x,batch_y) in enumerate(train_loader):
                
                epsilon = 1e-8  # Small constant for numerical stability
                iter_count += 1

                ################################# Derive Masks from Previous Run ###################################

                # Evaluate dropout probabilities based on loss differences for the final layer
                if self.current_epoch == 'ising':  # Ensure we are in the right phase

                    mask_list = []  # List to store masks for each layer in this model for the current batch
                    # Base list of substrings
                    # exclude_substrings = ['value', 'key', 'query', 'merge_layer']
                    exclude_substrings = ['merge_layer']
                    
                    # Dynamically add the 'decoder.decode_layers.i.linear_pred' substrings based on A_e
                    exclude_substrings += [f'decoder.decode_layers.{i}.linear_pred' for i in range(self.args.e_layers)]
                            
                    # Track the loss for the current model
                    original_loss = loss.item()  # Store the loss before modifying the weights
                    final_layer = self.models[0].classification_head2  # Assuming all models have the same final layer
    
                    # Calculate dropout probabilities for weights
                    weight_dropout_probs = []
                    for nr in range(final_layer.mean_weight.shape[0]):  # Loop through each output weight
                        for nc in range(final_layer.mean_weight.shape[1]):  # Loop through each input feature
                            original_weight = final_layer.mean_weight[nr, nc].item()
    
                            # Drop the weight temporarily
                            final_layer.mean_weight[nr, nc].data = torch.tensor(0.0, device=final_layer.mean_weight.device)
    
                            # Forward pass with the weight dropped
                            model = self.models[0]
                            
                            output_dropped,_ = self._process_one_batch(train_data, batch_x, batch_y, model)
                            output_dropped = output_dropped.float().squeeze()

                            # Compute loss for dropped case (penalized)
                            # loss_dropped = criterion(output_dropped, true)
        
                            # Calculate weights as the inverse of frequency
                            total_count = self.args.big_n
                            class_weights = [self.args.big_n / self.args.ctrl, self.args.big_n/(self.args.big_n-self.args.ctrl)]
                            class_weights = torch.tensor(class_weights, dtype=torch.float)

                            # compute unpenalized loss
                            # Initialize cross-entropy loss function
                            criterion_unp = nn.BCEWithLogitsLoss(pos_weight=class_weights[1] / class_weights[0])
                            # criterion_unp = nn.BCEWithLogitsLoss()
                            
                            # Compute cross-entropy loss
                            loss_dropped = -criterion(output_dropped, true) # Taking negative to convert to log likelihood instead of negative log-likelihood
                            loss_NoDrop = -criterion(pred, true)# Taking negative to convert to log likelihood instead of negative log-likelihood
                            
                            # Calculate loss difference
                            loss_difference = 0.5*(loss_NoDrop.item() - loss_dropped.item()) + epsilon # Compare with first model's loss
                            
                            # Calculate dropout probability
                            a_i = loss_difference
                            a_i_tensor = torch.tensor(a_i) if isinstance(a_i, float) else a_i
                            dropout_prob = 1 - 1 / (1 + torch.exp(-2*a_i_tensor))
                            weight_dropout_probs.append(dropout_prob)

                            # Restore the original weight
                            final_layer.mean_weight[nr, nc].data = torch.tensor(original_weight, device=final_layer.mean_weight.device)
    
                    # Convert list to tensor for easier manipulation
                    weight_dropout_probs = torch.tensor(weight_dropout_probs).to(final_layer.mean_weight.device)

                    mask = weight_dropout_probs.view(final_layer.mean_weight.shape).detach()
                    # mask = (mask - mask.min()) / (mask.max() - mask.min())
                    
                    mask_list.append(mask) # save masks for last layer in current batch
                    # Create mask based on computed dropout probabilities
                    # mask = torch.bernoulli(weight_dropout_probs.view(final_layer.mean_weight.shape))

                    # Pass the mask to the final layer for dropout
                    # final_layer.apply_custom_dropout(mask)
                    final_layer.apply_custom_dropout_prob(mask)
                    
                    layer = None  # Initialize variable to store the first layer
                    captured_layers = []  # List to store all layers that were already captured
                    layer_counter = 0

                    attention_triplet = {'query':0, 'key':0, 'value':0}
                    # First loop to find layers with 'mean_weight'
                    for name, param in reversed(list(model.named_parameters())):
                        
                        # Now use this list in your condition
                        if 'mean_weight' in name and all(substring not in name for substring in exclude_substrings):
                            if layer is None:  # Check if the first layer has not been found yet
                                layer_path = name.replace('.mean_weight', '')  # Capture the first layer
                                layer_names = layer_path.split('.')  # Split the path into parts
                                # Start from the model
                                layer = self.models[0]
                                
                                # Navigate through the attributes
                                for ln in layer_names:
                                    # Add check for nn.ModuleList or nn.Sequential to handle indexed layers
                                    if isinstance(layer, (nn.ModuleList, nn.Sequential)):
                                        if int(ln) < len(layer):
                                            layer = layer[int(ln)]  # Correctly handle indexing for layers like decode_layers[1]
                                    else:                            
                                        # Safeguard for getattr and debug info
                                        if hasattr(layer, ln):
                                            layer = getattr(layer, ln)
                                    
                                captured_layers.append(layer_path)  
                                
                            else:
                                next_layer_path = name.replace('.mean_weight', '')  # Capture the consecutive layer    
                                next_layer_names = next_layer_path.split('.')  # Split the path into parts
                                # Start from the model
                                next_layer = self.models[0]

                                # Navigate through the attributes
                                for ln in next_layer_names:
                                    if isinstance(next_layer, (nn.ModuleList, nn.Sequential)):
                                        if int(ln) < len(next_layer):
                                            next_layer = next_layer[int(ln)]  # Correctly handle indexing for layers like decode_layers[1]
                                    else:
                                        # Safeguard for getattr and debug info
                                        if hasattr(next_layer, ln):
                                            next_layer = getattr(next_layer, ln)
                                                    
                                ###########  Application Code here ########################
                                
                                # Compute connectivity based on the outgoing weights of the current layer
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

                                # if name in [f'decoder.decode_layers.{self.args.e_layers+1}.linear_pred.mean_weight' for i in range(self.args.e_layers+1)]:
                                if L_minus_1_connec.shape[0] == int(self.args.data_dim*self.args.d_model):
                                    plchld = L_minus_1_connec.shape[1]
                                    reshaped_tensor = L_minus_1_connec.t().reshape(plchld, -1, self.args.data_dim)  # Shape will be (batch_size, 20, 20)
                                    L_minus_1_connec = reshaped_tensor.mean(dim=2).t()  # Shape will be (1, 20)
                            
                                # Get the number of rows in the next layer's mean_weight
                                num_rows = next_layer.mean_weight.t().shape[0]
                                # Prepare the tensor for dropout probabilities

                                if 'l_pred_head.mean_weight' in name:
                                    a_i_tensor = torch.sum(L_minus_1_connec, dim=1).unsqueeze(0).repeat(num_rows, 1)
                                else:
                                    saliency_score = saliency_scores[name]
                                    # print(saliency_score.t().shape)
                                    # s_name = '.'.join(layer_names) + '.mean_weight'
                                    a_i_tensor = torch.sum(L_minus_1_connec, dim=1).unsqueeze(0).repeat(num_rows, 1) + 0.5*saliency_score.t()

                                L_minus_1_dropout_probs = 1 - 1 / (1 + torch.exp(-2 * a_i_tensor))
                                # Normalize
                                # L_minus_1_dropout_probs = (L_minus_1_dropout_probs - L_minus_1_dropout_probs.min()) / (L_minus_1_dropout_probs.max() - L_minus_1_dropout_probs.min())
                                # L_minus_1_dropout_probs = (L_minus_1_dropout_probs) / (L_minus_1_dropout_probs.max() + epsilon)

                                # if 'l_pred_head' in layer_names:
                                #     # Replicate each column once to get the shape (16, 20)
                                #     L_minus_1_dropout_probs = L_minus_1_dropout_probs.repeat(1, 2)  # Repeat the tensor 2 times along the second dimension


                                L_minus_1_dropout_probs = L_minus_1_dropout_probs.detach()
                                mask_list.append(L_minus_1_dropout_probs.t())  # Save masks for the current layer in the current batch
                                # print(L_minus_1_dropout_probs)
                                next_layer.apply_custom_dropout_prob(L_minus_1_dropout_probs.t())


                                if('value' in name):
                                    captured_layers.append(next_layer_path)
                                    attention_triplet['value'] = next_layer.mean_weight
                                elif('key' in name):
                                    captured_layers.append(next_layer_path)
                                    attention_triplet['key'] = next_layer.mean_weight
                                elif('query' in name):
                                    attention_triplet['query'] = next_layer.mean_weight
                                    # Update 'layer_name' to the latest found layer
                                    layer_path = next_layer_path  
                                    captured_layers.append(layer_path)  
                                    layer = next_layer  # Use getattr to dynamically access the layer
                                    layer_names = layer_path.split('.')  # Split the path into parts
                                else:
                                    # Update 'layer_name' to the latest found layer
                                    layer_path = next_layer_path  
                                    captured_layers.append(layer_path)  
                                    layer = next_layer  # Use getattr to dynamically access the layer
                                    layer_names = layer_path.split('.')  # Split the path into parts
                                
                                layer_counter += 1
                                ###########################################################
                                
                if self.current_epoch == 'ising':
                    batch_masks.append(mask_list)
                    if (i == (train_steps-1)) and (epoch == (self.args.train_epochs+self.args.ising_epochs-1)):
                        total_ones = 0  # Counter for total number of ones across all tensors                        
                        # Loop through each tensor in the mask_list
                        for fmask in mask_list:
                            # Apply the threshold: elements < 0.5 become 1, the rest become 0
                            final_mask = (fmask < 0.5).int()
                            # final_mask = 1-fmask
                            # Count the number of ones in the current tensor and add to the total
                            total_ones += torch.sum(final_mask).item()
                        print(f'Ising num params:{total_ones}')
                        self.ising_params = total_ones
    
                # if self.args.save_memory == True:
                #     batch_masks.clear()
                #####################################################################################################
                                
                # Zero the gradients for each model's optimizer
                for optimizer in model_optim:
                    optimizer.zero_grad()
                    
                # Loop over models and process batches
                shuffled_indices = list(range(len(self.models)))
                random.shuffle(shuffled_indices)

                # Loop over models randomly and process batches
                for model_idx in shuffled_indices:
                    model = self.models[model_idx]
                    pred, true = self._process_one_batch(train_data, batch_x, batch_y, model)  # Pass each model

                # for model_idx, model in enumerate(self.models):
                #     pred, true = self._process_one_batch(train_data, batch_x, batch_y, model)  # Pass each model
    
                    pred = pred.float().squeeze()
                    true = true.long().squeeze()
    
                    # Compute loss for each model
                    loss = criterion(pred, true)
                    train_loss[model_idx].append(loss.item())

                    # My Hessian implementation
                    if ((epoch == (self.args.train_epochs-1)) and (i == (train_steps-1))) or ((self.current_epoch == 'ising') and (i == (train_steps-1))):
                        saliency_scores = {}  # Dictionary to store saliency scores for each parameter
                        
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
                        optimizer.zero_grad()
                    
                    loss.backward(create_graph =True)  # Backpropagation

                    model_optim[model_idx].step()  # Update model

                #################################################################
                #################################################################
                
                if (i + 1) % 100 == 0:
                    avg_losses = [np.mean(losses) for losses in train_loss]
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss for each model: {avg_losses}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                #################################################################

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            # Validation and early stopping
            train_loss_avg = [np.mean(losses) for losses in train_loss]
            vali_loss_avg = [self.vali(vali_data, vali_loader, criterion, model) for model in self.models]  # Pass each model
            test_loss_avg = [self.vali(test_data, test_loader, criterion, model) for model in self.models]  # Pass each model

            # Store the masks for this epoch in the main epoch list
            self.epoch_mask_list.append(batch_masks)

            self.t_loss_tracker.append(train_loss_avg)
            self.v_loss_tracker.append(vali_loss_avg)
            self.s_loss_tracker.append(test_loss_avg)

            # print(f"Epoch: {epoch + 1}, Train Loss: {train_loss_avg}, Test Loss: {test_loss_avg}")
            
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss_avg}, Vali Loss: {vali_loss_avg}, Test Loss: {test_loss_avg}")
            early_stopping(vali_loss_avg, self.models, path)  # Early stopping on model[0] as reference
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

    def test(self, setting, save_pred = True, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()


        all_preds = [[] for _ in range(len(self.models))]
        all_trues = [[] for _ in range(len(self.models))]
        all_metrics = [[] for _ in range(len(self.models))]
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):

                batch_size = batch_x.shape[0]
                instance_num += batch_size
                
                # Process each model in self.models
                for model_idx, model in enumerate(self.models):
                    pred, true = self._process_one_batch(test_data, batch_x, batch_y, model, inverse)
                    pred = pred.float()
                    true = true.long()
                    
                    # Calculate metrics for each model
                    batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                    all_metrics[model_idx].append(batch_metric)
                    
                    if save_pred:
                        all_preds[model_idx].append(pred.detach().cpu().numpy())
                        all_trues[model_idx].append(true.detach().cpu().numpy())

        # result save
        folder_path = args.root_path + 'results/' + setting +'/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for model_idx in range(len(self.models)):
            metrics_all = np.stack(all_metrics[model_idx], axis=0)
            metrics_mean = metrics_all.sum(axis=0) / instance_num
    
            mae, mse, rmse, mape, mspe, lgls = metrics_mean
            print(f'Model {model_idx} - CBE: {lgls}')
    
            # Save metrics for the current model
            metrics_summ = np.array([mae, mse, rmse, mape, mspe, lgls])
            metrics_df = pd.DataFrame(metrics_summ, columns=['Metrics'])
            metrics_df.to_csv(folder_path + f'metrics_model_{model_idx}.csv', index=False)
    
            if save_pred:
                preds = np.concatenate(all_preds[model_idx], axis=0)
                trues = np.concatenate(all_trues[model_idx], axis=0)
    
                preds_df = pd.DataFrame(preds)
                preds_df.to_csv(folder_path + f'pred_model_{model_idx}.csv', index=False)
    
                trues_df = pd.DataFrame(trues)
                trues_df.to_csv(folder_path + f'true_model_{model_idx}.csv', index=False)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, model, inverse = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)  # Add a dimension to match output shape
    
        # Get model outputs
        outputs = model(batch_x)
    
        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred = True, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_gen(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = args.scale,
            scale_statistic = args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        metrics_all = [[] for _ in range(self.args.num_models)]  # List to hold metrics for each model
        all_preds = [[] for _ in range(self.args.num_models)]  # List to hold predictions for each model
        all_trues = [[] for _ in range(self.args.num_models)]  # List to hold true values for each model
        instance_num = 0

        # Evaluate each model
        for model_idx, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(data_loader):
                    pred, true = self._process_one_batch(
                        data_set, batch_x, batch_y, model, inverse)
    
                    pred = pred.float().squeeze()
                    true = true.long().squeeze()
    
                    batch_size = pred.shape[0]
                    instance_num += batch_size
                    batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                    metrics_all[model_idx].append(batch_metric)
    
                    if save_pred:
                        all_preds[model_idx].append(pred.detach().cpu().numpy())
                        all_trues[model_idx].append(true.detach().cpu().numpy())


        # result save
        folder_path = args.root_path + '/results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for model_idx in range(self.args.num_models):
            metrics_all[model_idx] = np.stack(metrics_all[model_idx], axis=0)
            metrics_mean = metrics_all[model_idx].sum(axis=0) / instance_num
    
            mae, mse, rmse, mape, mspe, lgls = metrics_mean
            print(f'Model {model_idx} - CBE: {lgls}')
    
            # Save metrics for the current model
            metrics_summ = np.array([mae, mse, rmse, mape, mspe, lgls])
            metrics_df = pd.DataFrame(metrics_summ, columns=['Metrics'])
            metrics_df.to_csv(folder_path + f'metrics_model_{model_idx}.csv', index=False)
    
            if save_pred:
                preds = np.concatenate(all_preds[model_idx], axis=0)
                trues = np.concatenate(all_trues[model_idx], axis=0)
    
                preds_df = pd.DataFrame(preds)
                preds_df.to_csv(folder_path + f'pred_model_{model_idx}.csv', index=False)
    
                trues_df = pd.DataFrame(trues)
                trues_df.to_csv(folder_path + f'true_model_{model_idx}.csv', index=False)
    
        return [mae, mse, rmse, mape, mspe, lgls]  # Return metrics for all models

    def Ising_test(self, setting, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()

        all_preds = [[] for _ in range(len(self.models))]

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                
                # Process each model in self.models
                for model_idx, model in enumerate(self.models):
                    pred, _ = self._process_one_batch(test_data, batch_x, batch_y, model, inverse)
                    pred = pred.float()

                    # Apply sigmoid to logits to get probabilities
                    pred_probs = torch.sigmoid(pred)
                    # Convert to class 0 or 1 based on threshold (usually 0.5)
                    pred_class = (pred_probs > 0.5).long()
    
                    # Store predictions and true labels
                    all_preds[model_idx].append(pred_class)                      

        return all_preds

    
    def Ising_test_train(self, setting, inverse = False):
        test_data, test_loader = self._get_data(flag='train')
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()

        all_preds = [[] for _ in range(len(self.models))]

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                
                # Process each model in self.models
                for model_idx, model in enumerate(self.models):
                    pred, _ = self._process_one_batch(test_data, batch_x, batch_y, model, inverse)
                    pred = pred.float()

                    # Apply sigmoid to logits to get probabilities
                    pred_probs = torch.sigmoid(pred)
                    # Convert to class 0 or 1 based on threshold (usually 0.5)
                    pred_class = (pred_probs > 0.5).long()
    
                    # Store predictions and true labels
                    all_preds[model_idx].append(pred_class)                      

        return all_preds


