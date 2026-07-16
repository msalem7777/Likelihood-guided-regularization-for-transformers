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

        # Cache std tensors when frozen (log_std is constant, exp() need not be recomputed per forward)
        self.freeze_std = freeze_std
        if freeze_std:
            self.register_buffer('_std_weight_cached', torch.exp(self.log_std_weight.detach()))
            if self.log_std_bias is not None:
                self.register_buffer('_std_bias_cached', torch.exp(self.log_std_bias.detach()))
        self.debug_checks = False

        # Epoch tracker to adjust behavior
        self.epoch_tracker = epoch_tracker
        # Placeholder for custom mask (set in training loop)
        self.custom_mask = None
        self.custom_mask_prob = None

    def _get_stds(self):
        if self.freeze_std:
            std_w = self._std_weight_cached
            std_b = self._std_bias_cached if self.log_std_bias is not None else None
        else:
            std_w = torch.exp(self.log_std_weight)
            std_b = torch.exp(self.log_std_bias) if self.log_std_bias is not None else None
        return std_w, std_b

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
        self.custom_mask = mask.to(self.mean_weight.device)
        
    # New method for applying custom dropout probability
    def apply_custom_dropout_prob(self, mask: torch.Tensor):
        """
        Apply a custom dropout mask to the weights.
        The mask should be the same shape as the mean_weight.
        """
        self.custom_mask_prob = mask.to(self.mean_weight.device)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        current_epoch = self.epoch_tracker.current_epoch if self.epoch_tracker else 'pilot'

        if self.training:
            std_weight, std_bias = self._get_stds()
            sampled_weights = std_weight * torch.randn_like(self.mean_weight)
            noisy_mean = self.mean_weight + std_weight * torch.randn_like(self.mean_weight)

            if self.custom_mask_prob is not None:
                p = self.custom_mask_prob
                if self.debug_checks and not torch.all(torch.isfinite(p)):
                    print(p)
                    raise RuntimeError("❌ custom_mask_prob contains NaNs or Infs")

                if current_epoch == "fine-tuning":
                    prob_mask = 1 - p
                    weight = noisy_mean * prob_mask + sampled_weights * (1 - prob_mask)
                else:
                    binary_mask = 1 - torch.bernoulli(p)
                    weight = noisy_mean * binary_mask + sampled_weights * (1 - binary_mask)
            else:
                if current_epoch == "fine-tuning":
                    prob_mask = 1 - self.p
                    weight = noisy_mean * prob_mask + sampled_weights * (1 - prob_mask)
                else:
                    binary_mask = 1 - torch.bernoulli(torch.full_like(self.mean_weight, self.p))
                    weight = noisy_mean * binary_mask + sampled_weights * (1 - binary_mask)

            if self.mean_bias is not None:
                bias = self.mean_bias + std_bias * torch.randn_like(self.mean_bias)
            else:
                bias = None
        else:
            # Evaluation: deterministic weighted means
            if self.custom_mask_prob is not None:
                weight = (1 - self.custom_mask_prob) * self.mean_weight
            else:
                weight = (1 - self.p) * self.mean_weight
            bias = self.mean_bias  # None if bias=False

        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return f'in_features={self.mean_weight.size(1)}, out_features={self.mean_weight.size(0)}, bias={self.mean_bias is not None}'
