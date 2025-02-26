#!/usr/bin/env python
# coding: utf-8

# In[17]:


from transformer_layers.bbb_linear import BBBLinear  # Import the custom BBBLinear layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Ensure that the dimension is divisible by the number of heads
        assert self.head_dim * heads == dim, "Embedding dimension must be divisible by number of heads"
        
        # Linear layers for queries, keys, and values (using BBBLinear)
        self.query = BBBLinear(dim, dim)
        self.key = BBBLinear(dim, dim)
        self.value = BBBLinear(dim, dim)
        
        # Output linear layer (using BBBLinear)
        self.out_projection = BBBLinear(dim, dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Linear transformations to get Q, K, V
        queries = self.query(query)
        keys = self.key(keys)
        values = self.value(values)
        
        # Split into multiple heads
        queries = queries.view(N, query_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        values = values.view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1))  # (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)  # (N, heads, query_len, key_len)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, values)  # (N, heads, query_len, head_dim)
        
        # Concatenate heads and pass through output layer
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.heads * self.head_dim)
        out = self.out_projection(out)
        
        return out

class TransformerEncoderLayerWithBBB(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(TransformerEncoderLayerWithBBB, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            BBBLinear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            BBBLinear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

class VisionTransformerWithBBB(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epoch_tracker=None
    ):
        super(VisionTransformerWithBBB, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.device = device
        self.epoch_tracker = epoch_tracker

        # Patch Embedding Layer
        self.patch_embedding = BBBLinear(3 * patch_size * patch_size, embed_dim)

        # Transformer Encoder
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayerWithBBB(
                    embed_dim, num_heads, mlp_ratio, dropout, device=device
                )
                for _ in range(depth)
            ]
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            BBBLinear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            BBBLinear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        # Move input to the correct device
        x = x.to(self.device)

        # Reshape and flatten patches
        batch_size = x.shape[0]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(batch_size, 3, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_patches, -1)

        # Apply patch embedding
        x = self.patch_embedding(patches)

        # Forward through transformer encoder layers
        for layer in self.encoder:
            x = layer(x)

        # Use CLS token for classification
        cls_token = x.mean(dim=1)

        # Forward through classification head
        logits = self.classification_head(cls_token)
        return logits

