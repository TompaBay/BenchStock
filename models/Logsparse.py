import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, LogSparseAttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.total_length = configs.pred_len + configs.seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LogSparseAttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, sparse_flag=configs.sparse_flag,
                                      win_len=configs.win_len, res_len=configs.res_len),
                        configs.d_model, configs.n_heads, configs.qk_ker, v_conv=configs.v_conv),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    LogSparseAttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, sparse_flag=configs.sparse_flag,
                                      win_len=configs.win_len, res_len=configs.res_len),
                        configs.d_model, configs.n_heads, configs.qk_ker, v_conv=configs.v_conv),
                    LogSparseAttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, sparse_flag=configs.sparse_flag,
                                      win_len=configs.win_len, res_len=configs.res_len),
                        configs.d_model, configs.n_heads, configs.qk_ker, v_conv=configs.v_conv),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.final = nn.Linear(configs.dec_in, configs.final_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **_):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out= self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return self.final(dec_out[:, -self.pred_len:, :]), attns
        else:
            return self.final(dec_out[:, -self.pred_len:, :])  # [B, L, D]