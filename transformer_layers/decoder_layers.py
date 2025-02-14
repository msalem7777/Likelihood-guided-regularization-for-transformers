import torch
import torch.nn as nn
from einops import rearrange
from .attention_layers import TwoStageAttentionLayer, AttentionLayer  # Importing from the attention layers module
from .bbblinear import BBBLinear  # Import BBBLinear from bbblinear.py

class DecoderLayer(nn.Module):
    '''
    The decoder layer of the Crossformer architecture, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, data_dim, d_ff=None, dropout=0.0, out_seg_num = 10, factor = 10, epoch_tracker=None):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)    
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(BBBLinear(d_model, d_model, epoch_tracker=epoch_tracker),
                                nn.GELU(),
                                BBBLinear(d_model, d_model, epoch_tracker=epoch_tracker))
        
        self.linear_pred = BBBLinear(d_model, seg_len, epoch_tracker=epoch_tracker) # (CHANGE) need to change final output "seg_len" to match number of classes 
        

    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''

        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp = self.cross_attention(
            x, cross, cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num d_model -> b (out_d seg_num) d_model')

        return dec_output, layer_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, data_dim, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10, epoch_tracker=None):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, data_dim, d_ff, dropout,\
                                        out_seg_num, factor, epoch_tracker=epoch_tracker))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            
            x, layer_predict = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict