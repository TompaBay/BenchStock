from torch_geometric import nn
import torch
import torch.nn
import torch.nn.functional as F
 

class Attention(torch.nn.Module):
    def __init__(self, args, dimensions, query_len, device):
        super(Attention, self).__init__()
        self.dim = dimensions
        self.attn = torch.nn.MultiheadAttention(dimensions, 4, batch_first=True)
        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions)
        self.gamma = torch.nn.Parameter(torch.FloatTensor(args.n_stocks, 1, 1)) 
        self.epsilon = torch.nn.Parameter(torch.FloatTensor(args.n_stocks, 1, 1))
        self.gamma.requires_grad = True
        self.epsilon.requires_grad = True
        self.device = device

    def forward(self, query, key, value):
        query_len = key.shape[1]
        z, attention_weights = self.attn(query, key, value)
        
        #Hawkes Attention
        #lambda
        lam = (attention_weights.permute(0, 2, 1).repeat(1, 1, self.dim)) * value

        #delta
        delta_t = torch.flip(torch.arange(1, query_len + 1), [0]).type(torch.float32).to(self.device)
        delta_t = delta_t.reshape(1, -1, 1).repeat(lam.shape[0], 1, lam.shape[2])

        # e^ -gamma delta
        decay = torch.exp(-1 * self.gamma * delta_t)
        hawkes= F.relu(self.epsilon * lam * decay)
        #z
        z = torch.sum(lam + hawkes, dim=1, keepdim=True)

        combined = torch.cat((z, query), dim=2)
        output = self.linear_out(combined)
        return output


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.in_dim = args.enc_in
        self.out_dim = args.d_model
        self.gru = torch.nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, batch_first=True)
        self.lstm = torch.nn.LSTM(self.in_dim, self.out_dim, batch_first=True) 
        self.lstm1 = torch.nn.LSTMCell(self.in_dim, self.out_dim)
        self.key_net = torch.nn.Linear(self.out_dim, self.out_dim)
        self.value_net = torch.nn.Linear(self.out_dim, self.out_dim)
        self.attention = Attention(args, self.out_dim, args.seq_len - 1, self.args.gpu)
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.e_layers):
            self.layers.append(nn.HypergraphConv(self.out_dim, self.out_dim, use_attention=True, heads=self.args.n_heads, concat=False, negative_slope=self.args.negative_slope, dropout=self.args.dropout, bias=True))

        self.activation = torch.nn.LeakyReLU(self.args.negative_slope)
        self.linear = torch.nn.Linear(self.out_dim, args.c_out)
    
    
    def forward(self, input_feature, inci_mat, edge):
        output, hidden = self.lstm(input_feature[:, :-1])
        key = self.key_net(output)
        value = self.value_net(output)
        hidden_0 = self.lstm1(input_feature[:, -1], (hidden[0][0], hidden[1][0]))
        query = hidden_0[0]
        output = self.attention(query.unsqueeze(dim=1), key, value)
        output = output.squeeze(dim=1)

        edge_attr = torch.matmul(inci_mat.t(), output)
        for layer in self.layers:
            output = self.activation(layer(output, edge, hyperedge_attr=edge_attr))
        output = self.linear(output)
        return output
