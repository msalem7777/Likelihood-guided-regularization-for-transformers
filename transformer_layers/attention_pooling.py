import torch
import torch.nn as nn
from .bbb_linear import BBBLinear

class AttentionPooling(nn.Module):
    def __init__(self, d_model, epoch_tracker=None):
        super().__init__()
        self.attention_weights = BBBLinear(d_model, 1, epoch_tracker=epoch_tracker)

    def forward(self, x):
        # x: (batch_size, data_dim, num_segments, d_model)
        attn_scores = self.attention_weights(x).softmax(dim=2)  # Shape: (batch_size, data_dim, num_segments, 1)
        pooled = (attn_scores * x).sum(dim=2)  # Shape: (batch_size, data_dim, d_model)
        return pooled