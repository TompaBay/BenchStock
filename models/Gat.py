import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.hidden_size = args.d_model
        self.d_feat = args.enc_in
        self.rnn = nn.LSTM(input_size=self.d_feat, hidden_size=self.hidden_size, num_layers=args.e_layers, batch_first=True)
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.args.c_out)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)


    def cal_attention(self, x, y, adj_mat):
        x = self.transformation(x)
        y = self.transformation(y)
        
        # x shape: (sample_num, sample_num)
        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        attention_out *= adj_mat
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x, adj_mat, edge=None):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        for i in range(self.args.d_layers):
            att_weight = self.cal_attention(hidden, hidden, adj_mat)
            hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        
        return self.fc_out(hidden)