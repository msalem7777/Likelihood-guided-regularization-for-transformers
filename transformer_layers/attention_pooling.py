"""DEPRECATED. Unused by the current ViT pipeline.

AttentionPooling is retained for reuse with alternate architectures; it is not
imported by main/VisionTransformer_Trainer.py. Kept for provenance and may be
removed in a future cleanup.
"""
import warnings

import torch
import torch.nn as nn
from .bbb_linear import BBBLinear


class AttentionPooling(nn.Module):
    def __init__(self, d_model, epoch_tracker=None):
        super().__init__()
        warnings.warn(
            "AttentionPooling is deprecated and unused by the current ViT pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.attention_weights = BBBLinear(d_model, 1, epoch_tracker=epoch_tracker)

    def forward(self, x):
        # x: (batch_size, data_dim, num_segments, d_model)
        attn_scores = self.attention_weights(x).softmax(dim=2)
        pooled = (attn_scores * x).sum(dim=2)
        return pooled