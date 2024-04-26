import torch
import torch.nn as nn

class Model(nn.Module):
    # define model elements
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layers = []
        
        self.layers.append(nn.Linear(configs.enc_in * configs.seq_len, configs.hidden_layer[0]))
        for i in range(len(configs.hidden_layer) - 1):
            self.layers.append(nn.BatchNorm1d(configs.hidden_layer[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(configs.hidden_layer[i], configs.hidden_layer[i+1]))

        self.layers.append(nn.BatchNorm1d(configs.hidden_layer[-1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(configs.hidden_layer[-1], configs.c_out))
        
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x, test=False):
        return self.model(x)