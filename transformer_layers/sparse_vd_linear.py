"""Sparse Variational Dropout linear layer.

This is a small PyTorch port of the ``LinearSVDO`` layer in the paper-author
reference implementation for Molchanov, Ashukha, and Vetrov (ICML 2017).
The implementation keeps the paper's three defining choices:

* one learned posterior variance per weight;
* local reparameterization during training, which samples output activations
  instead of constructing one noisy weight matrix per minibatch; and
* deterministic pruning when ``log(alpha)`` exceeds 3 at evaluation time.

Biases remain deterministic, matching the reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseVDLinear(nn.Module):
    """Linear layer trained with per-weight Sparse Variational Dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 3.0,
        log_sigma_init: float = -5.0,
        log_alpha_clip: float = 8.0,
        train_clip: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.threshold = float(threshold)
        self.log_sigma_init = float(log_sigma_init)
        self.log_alpha_clip = float(log_alpha_clip)
        self.train_clip = bool(train_clip)

        if self.log_alpha_clip <= self.threshold:
            raise ValueError(
                "log_alpha_clip must exceed the pruning threshold; "
                f"received clip={self.log_alpha_clip}, threshold={self.threshold}"
            )

        self.mean_weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.log_sigma_weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if bias:
            self.mean_bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("mean_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use Lasagne's ``Normal()`` default from the paper-author code."""
        nn.init.normal_(self.mean_weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sigma_weight, self.log_sigma_init)
        if self.mean_bias is not None:
            nn.init.zeros_(self.mean_bias)

    def log_alpha(self) -> torch.Tensor:
        """Return the clipped log noise-to-signal ratio for every weight."""
        value = 2.0 * self.log_sigma_weight - 2.0 * torch.log(
            self.mean_weight.abs() + 1e-16
        )
        return torch.clamp(
            value,
            min=-self.log_alpha_clip,
            max=self.log_alpha_clip,
        )

    def posterior_variance_weight(self) -> torch.Tensor:
        """Return ``exp(clipped log(alpha)) * mean_weight**2``."""
        return torch.exp(self.log_alpha()) * self.mean_weight.square()

    def retained_weight_mask(self) -> torch.Tensor:
        """Return the reference keep mask: ``log(alpha) < threshold``."""
        return self.log_alpha() < self.threshold

    def deterministic_weight(self) -> torch.Tensor:
        """Return posterior means after applying the canonical pruning rule."""
        return self.mean_weight * self.retained_weight_mask().to(self.mean_weight.dtype)

    def training_mean_weight(self) -> torch.Tensor:
        """Return the dense-author training mean, with optional threshold clipping."""
        if self.train_clip:
            return self.deterministic_weight()
        return self.mean_weight

    def posterior_weight_sample(self) -> torch.Tensor:
        """Draw one posterior weight sample and zero weights marked for pruning."""
        posterior_std = torch.sqrt(self.posterior_variance_weight() + 1e-16)
        sampled = self.mean_weight + posterior_std * torch.randn_like(self.mean_weight)
        return sampled * self.retained_weight_mask().to(sampled.dtype)

    def kl_divergence(self) -> torch.Tensor:
        """Return the paper's differentiable approximation to ``KL(q(w)||p(w))``."""
        log_alpha = self.log_alpha()
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        constant = -k1

        # Exact approximation used in the attached paper-author repository.
        negative_kl = k1 * torch.sigmoid(k2 + k3 * log_alpha)
        negative_kl = negative_kl - 0.5 * F.softplus(-log_alpha)
        negative_kl = negative_kl + constant
        return -negative_kl.sum()

    def sparsity_stats(self) -> tuple[int, int]:
        """Return ``(pruned_weights, total_weights)`` using the threshold rule."""
        retained = self.retained_weight_mask()
        total = retained.numel()
        pruned = total - int(retained.sum().item())
        return pruned, total

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Local reparameterization: compute the exact mean and variance of
            # each output activation, then sample one activation-level epsilon.
            output_mean = F.linear(input, self.training_mean_weight(), self.mean_bias)
            output_variance = F.linear(
                input.square(),
                self.posterior_variance_weight(),
                bias=None,
            )
            output_std = torch.sqrt(output_variance + 1e-8)
            return output_mean + output_std * torch.randn_like(output_std)

        return F.linear(input, self.deterministic_weight(), self.mean_bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.mean_bias is not None}, threshold={self.threshold:g}, "
            f"log_alpha_clip={self.log_alpha_clip:g}, train_clip={self.train_clip}"
        )
