import torch
import torch.nn as nn
import torch.nn.functional as F
 

import torch
import torch.nn as nn

class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate weights
        # self.Wr = nn.Parameter(torch.Tensor(hidden_size, input_size))
        # self.Ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.br = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.Wr = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.Linear(input_size, hidden_size)
        self.br = nn.Parameter(torch.Tensor(1, hidden_size))
        
        # Update gate weights
        self.Wz = nn.Linear(hidden_size, hidden_size)
        self.Uz = nn.Linear(input_size, hidden_size)
        self.bz = nn.Parameter(torch.Tensor(1, hidden_size))
        
        # Candidate output weights
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Uh = nn.Linear(input_size, hidden_size)
        self.bh = nn.Parameter(torch.Tensor(1, hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)
    
    def sigmoid(self, x):
        return torch.sigmoid(x)
    
    def tanh(self, x):
        return torch.tanh(x)
    
    def forward(self, x, h):
        # Reset gate
        r = self.sigmoid(self.Wr(x) + self.Ur(h) + self.br)
        
        # Update gate
        z = self.sigmoid(self.Wz(x) + self.Uz(h) + self.bz)

        # Candidate output
        h_hat = self.tanh(self.Uh(r * h) + self.Wh(x) + self.bh)
        
        # Update hidden state
        h = (1 - z) * h + z * h_hat
        
        return h


class AggregationLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AggregationLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_matrix):
        output = self.linear(x)
        B, T, D = output.shape
        output = torch.mm(adj_matrix, output.reshape((B, -1))).reshape((B, T, D))
        return output


class TimeAdaptiveAttention(nn.Module):
    def __init__(self, in_features, heads, df):
        super(TimeAdaptiveAttention, self).__init__()
        self.in_features = in_features
        self.heads = heads
        self.df = df

        # Parameters for temporal distance rescaling
        self.ah = nn.Parameter(torch.randn(1))  # Weight parameter ah
        self.vh = nn.Parameter(torch.randn(1))  # Upperbound and steepness parameter vh

        # Linear transformations for attention computation
        self.linear_q = nn.Linear(in_features, df * heads)
        self.linear_k = nn.Linear(in_features, df * heads)
        self.linear_v = nn.Linear(in_features, df * heads)

        # Additional transformation matrix for attention computation
        self.w_q_cross = nn.Linear(in_features, df * heads)
        self.w_k_cross = nn.Linear(in_features, df * heads)
        self.w_v_cross = nn.Linear(in_features, df * heads)

    def forward(self, input_seq, device):
        batch_size, seq_len, _ = input_seq.size()

        z = torch.arange(1, seq_len + 1).unsqueeze(0).repeat(seq_len, 1)
        # Compute relative temporal distances
        distances = torch.abs(z - z.T).to(device)

        # Apply time-adaptive sigmoid function to rescale distances
        z_hat = 1 + torch.exp(self.vh) / (1 + torch.exp(self.vh - self.ah * distances))

        # Linear transformations
        keys = self.linear_k(input_seq).view(batch_size, seq_len, self.heads, self.df).transpose(1, 2)
        values = self.linear_v(input_seq).view(batch_size, seq_len, self.heads, self.df).transpose(1, 2)
        query = self.linear_q(input_seq).view(batch_size, seq_len, self.heads, self.df).transpose(1, 2)

        # # Compute cross patterns
        # cross_query = self.w_q_cross(input_seq)
        # cross_keys = self.w_k_cross(input_seq)
        # cross_values = self.w_v_cross(input_seq)

        # cross_q = cross_query.view(batch_size, seq_len, self.heads, self.df)
        # cross_k = cross_keys.view(batch_size, seq_len, self.heads, self.df)
        # cross_v = cross_values.view(batch_size, seq_len, self.heads, self.df)

        # Compute attention scores
        attention_scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_scores *= z_hat.unsqueeze(0) / (self.df ** 0.5)
        attention_weights = F.relu(F.softmax(attention_scores, dim=-1))

    
        # # Apply element-wise product with cross patterns
        # cross_product = cross_q * cross_k
        # cross_product *= z_hat.unsqueeze(0) / (self.df ** 0.5)
        # cross_attention = F.relu(cross_product) * attention_weights.unsqueeze(-2)
        cross_values = torch.matmul(attention_weights, values).reshape(batch_size, seq_len, -1)

        return cross_values



class LocalInteractionLayer(nn.Module):
    def __init__(self, in_features, out_features, w, df, heads):
        super(LocalInteractionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = w
        self.df = df
        self.heads = heads

        self.linear_query = nn.Linear(in_features, df * heads)
        self.linear_key = nn.Linear(in_features, df * heads)
        self.linear_value = nn.Linear(in_features, df * heads)

    def forward(self, input_seq, device):
        batch_size, seq_len, _ = input_seq.size()
        
        # Initialize lists to store local compound patterns
        compound_patterns = []

        for tau in range(seq_len):
            # Get the local context for each time-step tau
            context = input_seq[:, max(0, tau - self.w + 1):tau + 1, :]  # [batch_size, w, in_features]
            
            # Pad with zeros if needed for initial time steps
            if context.size(1) < self.w:
                pad = torch.zeros(batch_size, self.w - context.size(1), self.in_features).to(device)
                context = torch.cat((pad, context), dim=1)

            # Represent transformed key-value tuples
            keys = self.linear_key(context).view(batch_size, self.heads, self.w, self.df)  # [batch_size, heads, w, df]
            values = self.linear_value(context).view(batch_size, self.heads, self.w, self.df)  # [batch_size, heads, w, df]

            # Form the query matrix
            query = self.linear_query(input_seq[:, tau, :]).unsqueeze(1)  # [batch_size, 1, df * heads]
            query = query.view(batch_size, self.heads, 1, self.df)  # [batch_size, heads, 1, df]

            # Compute attention scores
            attention_scores = torch.matmul(query, keys.transpose(2, 3)) / (self.df ** 0.5)  # [batch_size, heads, 1, w]

            # Apply softmax
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, heads, 1, w]

            # Compute compound pattern using attention weights
            compound_pattern = torch.matmul(attention_weights, values)  # [batch_size, heads, 1, df]

            # Concatenate all heads and reshape
            compound_pattern = compound_pattern.view(batch_size, -1)  # [batch_size, heads * df]
            
            compound_patterns.append(compound_pattern)

        # Concatenate all time-steps to get the final output of this layer
        compound_patterns = torch.stack(compound_patterns, dim=1)  # [batch_size, seq_len, heads * df]

        return compound_patterns
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class Model(nn.Module):
    def __init__(self, args):
                #  in_dim, hid_dim, out_dim, num_layer, ws, device, head=6):
        super(Model, self).__init__()
        self.in_dim = args.enc_in
        self.out_dim = args.d_model
        # self.device = device
        self.embedding = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_layers = nn.ModuleList()
        
        self.graph_layers.append(AggregationLayer(self.in_dim, self.hid_dim))
        self.graph_layers.append(GRULayer(self.hid_dim, self.hid_dim))

        for i in range(args.e_layers - 1):
            self.graph_layers.append(AggregationLayer(self.hid_dim, self.hid_dim))
            self.graph_layers.append(GRULayer(self.hid_dim, self.hid_dim))
        
        self.Lit_layers = []
        df = int(self.hid_dim // args.head)
        for w in ws:
            self.Lit_layers.append(LocalInteractionLayer(hid_dim, hid_dim, w, df, head).to(device))
            self.Lit_layers.append(TimeAdaptiveAttention(hid_dim, head, df).to(device))
            self.Lit_layers.append(FeedForwardNetwork(hid_dim, hid_dim).to(device))

        self.norm = nn.LayerNorm(hid_dim)
        self.linear = nn.Linear(hid_dim * len(ws), out_dim)
        self.leaky = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x, adj_matrix, edge=None):
        a = self.graph_layers[0](x, adj_matrix)
        e = self.embedding(x)
        e = self.graph_layers[1](a, e)
        for i in range(2, len(self.graph_layers), 2):
            a = self.graph_layers[i](e, adj_matrix)
            e = self.graph_layers[i+1](a, e)
        
        output = e
        F_list = []
        for i in range(0, len(self.Lit_layers), 3):
            output = self.Lit_layers[i](output, self.device)
            F = self.Lit_layers[i+1](output, self.device)
            F = self.Lit_layers[i+2](F)
            F = self.norm(output + F)
            F_list.append(F)
        
        output = torch.mean(torch.stack(F_list, dim=0), dim=2).transpose(0, 1).reshape(x.shape[0], -1)
        ret = self.leaky(self.linear(output))

        return output, ret
