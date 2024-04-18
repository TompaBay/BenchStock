import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        self.dim = dimensions
        self.attn = nn.MultiheadAttention(dimensions, 1, batch_first=True)
        self.linear_out = nn.Linear(dimensions * 2, dimensions)

    def forward(self, query, key, value):
        output, attention_weights = self.attn(query, key, value)
        output = torch.cat((output, query), dim=2)
        output = self.linear_out(output)
        return output, attention_weights



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc = nn.LSTM(input_size=configs.enc_in, hidden_size=configs.d_model, num_layers=configs.e_layers, batch_first=True, dropout=configs.dropout) 
        self.dec = nn.LSTM(input_size=configs.dec_in, hidden_size=configs.d_model, num_layers=configs.d_layers, batch_first=True, dropout=configs.dropout)

        self.key_net = nn.Linear(configs.d_model, configs.d_model)
        self.value_net = nn.Linear(configs.d_model, configs.d_model)
        self.attention = Attention(configs.d_model)
        self.linear = nn.Linear(configs.d_model, configs.c_out)
        
    
    def forward(self, x, test=False):
        output, hidden = self.enc(x[:, :-1])
        key = self.key_net(output)
        value =self.value_net(output)

        query, hidden  = self.dec(x[:, -1:], hidden)
        output, attn = self.attention(query, key, value)
        output = self.linear(output)

        return output
