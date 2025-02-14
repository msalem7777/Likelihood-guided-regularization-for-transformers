import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange
from .bbblinear import BBBLinear  # Now importing BBBLinear from bbblinear.py

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.0, scores_on=None):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.scores_on = scores_on

        # # Initialize a list to store the scores
        # self.scores_list = []
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        
        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1, scores_on = None, epoch_tracker=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout, scores_on = scores_on)
        self.query_projection = BBBLinear(d_model, d_keys * n_heads, epoch_tracker=epoch_tracker)
        self.key_projection = BBBLinear(d_model, d_keys * n_heads, epoch_tracker=epoch_tracker)
        self.value_projection = BBBLinear(d_model, d_values * n_heads, epoch_tracker=epoch_tracker)
        self.out_projection = BBBLinear(d_values * n_heads, d_model, epoch_tracker=epoch_tracker)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.0, epoch_tracker=None):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout, scores_on='time', epoch_tracker=epoch_tracker)
        self.dim_attention = AttentionLayer(d_model, n_heads, dropout = dropout, scores_on='cross', epoch_tracker=epoch_tracker)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(BBBLinear(d_model, d_ff, epoch_tracker=epoch_tracker),
                                nn.GELU(),
                                BBBLinear(d_ff, d_model, epoch_tracker=epoch_tracker))
        self.MLP2 = nn.Sequential(BBBLinear(d_model, d_ff, epoch_tracker=epoch_tracker),
                                nn.GELU(),
                                BBBLinear(d_ff, d_model, epoch_tracker=epoch_tracker))

    def forward(self, x):
        #Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
       
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)

        # Apply cross-dimensional attention
        dim_enc = self.dim_attention(dim_send, dim_send, dim_send)

        # Add residual and apply normalization & MLP
        dim_enc = dim_send + self.dropout(dim_enc)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)

        return final_out

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed
