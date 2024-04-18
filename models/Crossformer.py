import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.cross_embed import DSW_embedding

from math import ceil

class Model(nn.Module):
    def __init__(self, configs, baseline=False):
        super(Model, self).__init__()
        self.baseline = baseline

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * configs.seq_len / configs.seg_len) * configs.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / configs.seg_len) * configs.seg_len
        self.in_len_add = self.pad_in_len - configs.seq_len
        self.out_len = configs.pred_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(configs.seg_len, configs.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, (self.pad_in_len // configs.seg_len), configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(configs.e_layers, configs.win_size, configs.d_model, configs.n_heads,configs.d_ff, block_depth = 1, \
                                    dropout=configs.dropout, in_seg_num=(self.pad_in_len // configs.seg_len), factor=configs.factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, (self.pad_out_len // configs.seg_len), configs.d_model))
        self.decoder = Decoder(configs.seg_len, configs.e_layers + 1, configs.d_model, configs.n_heads, configs.d_ff, configs.dropout, \
                                    out_seg_num = (self.pad_out_len // configs.seg_len), factor=configs.factor)
        
        self.final = nn.Linear(configs.enc_in, configs.c_out)
        
        
    def forward(self, x_seq, test=False):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)
        
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        
        return self.final(base + predict_y[:, :self.out_len, :]).squeeze(-1)