"""DEPRECATED. Unused by the current ViT pipeline.

get_sinusoidal_embeddings provides fixed sinusoidal positional embeddings. The
current ViT learns its patch embedding instead, so this is retained only for
reuse. Kept for provenance and may be removed in a future cleanup.
"""
import warnings

import torch
import numpy as np


def get_sinusoidal_embeddings(data_dim, seq_len, d_model):
    warnings.warn(
        "get_sinusoidal_embeddings is deprecated and unused by the current ViT pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    pe = torch.zeros(data_dim, seq_len, d_model)
    for i in range(data_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[i, :, 0::2] = torch.sin(position * div_term)
        pe[i, :, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, data_dim, seq_len, d_model)