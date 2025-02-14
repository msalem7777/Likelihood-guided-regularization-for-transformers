class Transformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                factor=10, d_model=32, d_ff=64, n_heads=8, e_layers=2, 
                dropout=0.0, baseline=False, device=torch.device('cuda:0'), num_classes=2, epoch_tracker=None):
        super(Transformer, self).__init__()  # Corrected here
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len  # (CHANGE) Take out this variable
        self.seg_len = seg_len
        self.merge_win = win_size
        self.num_classes = num_classes

        self.baseline = baseline
        self.device = device

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len  # (CHANGE) Take out
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        # self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = get_sinusoidal_embeddings(data_dim, (self.pad_in_len // seg_len), d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))  # Old Random 
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1, 
                                    dropout=dropout, in_seg_num=(self.pad_in_len // seg_len), factor=factor, epoch_tracker=epoch_tracker)

        self.attention_pooling = AttentionPooling(d_model)
        # Classification head
        self.classification_head = nn.Sequential(
            BBBLinear(d_model * data_dim, d_model // 2, epoch_tracker=epoch_tracker),
            nn.GELU(),
            BBBLinear(d_model // 2, d_model // 4, epoch_tracker=epoch_tracker)
        )

        self.classification_head2 = BBBLinear(d_model // 4, 1, epoch_tracker=epoch_tracker)
        
        # Other initializations...
        self.epoch_tracker = epoch_tracker  # Track epochs for adjusting behaviors
        
    def forward(self, x_seq):
        x_seq = x_seq.to(self.device)
        
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)
        enc_out = enc_out[-1]  # Select the last block's output

        ###########  OPTION 1: MEAN #############################
        # blinear = enc_out.mean(dim=2)
        # blinear = blinear.reshape(blinear.shape[0], -1)
        ###########  OPTION 2: MAX ##############################
        # blinear,_ = enc_out.max(dim=2)
        ###########  OPTION 3: FLATTEN ##########################
        # blinear = enc_out.reshape(enc_out.shape[0], -1)
        ###########  OPTION 4: ATTENTION ########################
        enc_out = self.attention_pooling(enc_out)  # Shape: (batch_size, data_dim, d_model)
        blinear = enc_out.reshape(enc_out.shape[0], -1)  # Shape: (batch_size, data_dim * d_model)
        
        features = self.classification_head(blinear)
        logits = self.classification_head2(features)

        return logits
