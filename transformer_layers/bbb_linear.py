import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining Bayesian Linear MLP layers
class BBBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, p: float = 0.0, bias: bool = True, freeze_std: bool = True, device=None, dtype=None, epoch_tracker=None) -> None:
        super(BBBLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Learnable mean for the weight matrix
        self.mean_weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Learnable standard deviation for the weight matrix
        self.log_std_weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # Mixture probability p (the probability to drop a weight)
        self.p = p
        
        if bias:
            # Learnable mean for the bias term
            self.mean_bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.log_std_bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))  # log standard deviation for bias
        else:
            self.register_parameter('mean_bias', None)
            self.register_parameter('log_std_bias', None)
        
        self.reset_parameters()

        # Optionally freeze the standard deviation by disabling its gradient
        if freeze_std:
            self.log_std_weight.requires_grad = False
            if self.log_std_bias is not None:
                self.log_std_bias.requires_grad = False

        # Epoch tracker to adjust behavior
        self.epoch_tracker = epoch_tracker
        # Placeholder for custom mask (set in training loop)
        self.custom_mask = None
        self.custom_mask_prob = None

    def reset_parameters(self) -> None:
        # Initialize mean weights and biases using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.mean_weight, a=math.sqrt(5))
        if self.mean_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mean_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mean_bias, -bound, bound)
        
        # Initialize the log standard deviations to small values (e.g., log(0.01))
        nn.init.constant_(self.log_std_weight, -4.0)  # Small initial std deviation
        if self.mean_bias is not None:
            nn.init.constant_(self.log_std_bias, -4.0)

    # New method for applying custom dropout
    def apply_custom_dropout(self, mask: torch.Tensor):
        """
        Apply a custom dropout mask to the weights.
        The mask should be the same shape as the mean_weight.
        """
        self.custom_mask = mask
        
    # New method for applying custom dropout probability
    def apply_custom_dropout_prob(self, mask: torch.Tensor):
        """
        Apply a custom dropout mask to the weights.
        The mask should be the same shape as the mean_weight.
        """
        self.custom_mask_prob = mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = self.mean_weight.device
        current_epoch = self.epoch_tracker.current_epoch if self.epoch_tracker else 'pilot'
            
        # Sample standard deviation for the weights
        std_weight = torch.exp(self.log_std_weight).to(device)
        
        if self.training:   
            # Sample from the normal distribution centered at the mean weight            
            sampled_weights = torch.normal(mean=torch.zeros_like(self.mean_weight, device=device),std=std_weight)

            if self.custom_mask_prob is not None:
                # Apply the custom mask (if provided) to the mean_weight

                p = self.custom_mask_prob
                if not torch.all(torch.isfinite(p)):
                    print(p)
                    raise RuntimeError("❌ custom_mask_prob contains NaNs or Infs")

                m_min, m_max, m_mean = p.min().item(), p.max().item(), p.mean().item()
                # print(f"[Mask Check] mask ∈ [{m_min:.4f}, {m_max:.4f}], mean = {m_mean:.4f}")

                if m_mean > 0.99:
                    print("⚠️ Mask is nearly all 1s — might be dropping everything")
                elif m_mean < 0.01:
                    print("⚠️ Mask is nearly all 0s — might be keeping everything")

                binary_mask = 1-torch.bernoulli(self.custom_mask_prob.to(device).view(self.custom_mask_prob.shape))
                prob_mask = 1-self.custom_mask_prob.to(device)
                if current_epoch == "fine-tuning":
                    weight = (self.mean_weight + std_weight * torch.randn_like(self.mean_weight, device=device)) * prob_mask + sampled_weights * (1 - prob_mask)
                else:
                    weight = (self.mean_weight + std_weight * torch.randn_like(self.mean_weight, device=device)) * binary_mask + sampled_weights * (1 - binary_mask)

            else:
                # Create a binary mask to apply DropConnect (randomly keep or "drop" weights)
                binary_mask = 1 - torch.bernoulli(torch.full_like(self.mean_weight, self.p))
                prob_mask = 1-self.p
                if current_epoch == "fine-tuning":
                    weight = (self.mean_weight + std_weight * torch.randn_like(self.mean_weight, device=device)) * prob_mask + sampled_weights * (1 - prob_mask)        
                else:
                    weight = (self.mean_weight + std_weight * torch.randn_like(self.mean_weight, device=device)) * binary_mask + sampled_weights * (1 - binary_mask)        
        else:
            # In evaluation mode, use weighted means
            # mvn_0 = torch.normal(0, std_weight).to(device)
            # mvn_M = self.mean_weight + std_weight * torch.randn_like(self.mean_weight, device=device)
        
            # Apply custom mask during evaluation if available
            if self.custom_mask_prob is not None:
                weight =  (1 - self.custom_mask_prob.to(device)) * self.mean_weight
            else:
             # Weighted sum based on p
                weight = (1 - self.p) * self.mean_weight   

        # Handle bias similarly
        if self.mean_bias is not None:
            if self.training:
                
                std_bias = torch.exp(self.log_std_bias).to(device)
                bias = (self.mean_bias + std_bias * torch.randn_like(self.mean_bias, device=device))
                
            else:
                bias = self.mean_bias
        else:
            bias = None
                
        # Apply the linear transformation using the masked weights and bias
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return f'in_features={self.mean_weight.size(1)}, out_features={self.mean_weight.size(0)}, bias={self.mean_bias is not None}'
