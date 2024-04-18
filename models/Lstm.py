import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=configs.d_model,num_layers=configs.e_layers, batch_first=True, dropout=configs.dropout) 
        self.linear = nn.Linear(configs.d_model, configs.c_out)
    
    
    def forward(self, x, test=False):
        output, (h_n, c_n) = self.lstm(x)
        output = self.linear(output).squeeze(dim=-1)
        if test:
            output = output[:, -1:]
        return output
