import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.Linear = nn.Linear(input_dim, hidden_dim)
        self.ReLU = nn.LeakyReLU(0.2)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        output = self.Linear(x)
        output = self.ReLU(output)
        output, h_n = self.GRU(output)
        return output, h_n
    

class Alpha(nn.Module):
    def __init__(self, input_dim, output_dim, y_dim):
        super(Alpha, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_mu = nn.Linear(output_dim, y_dim)
        self.linear_sigma = nn.Linear(output_dim, y_dim)

    def forward(self, x):
        h = F.leaky_relu(self.linear(x), 0.2)
        mu = self.linear_mu(h)
        sigma = F.softplus(self.linear_sigma(h))
        return mu, sigma
    

class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureDecoder, self).__init__()
        self.alpha = Alpha(input_dim, output_dim, 1)
        self.beta = nn.Linear(input_dim, output_dim)


    def forward(self, e, Z_mu, Z_sigma):
        alpha_mu, alpha_sigma = self.alpha(e)
        beta_load = self.beta(e)
        Z_mu, Z_sigma = Z_mu.unsqueeze(1), Z_sigma.unsqueeze(1)
        y_mu = alpha_mu.squeeze(-1) + torch.sum(beta_load * Z_mu, dim=2)
        y_sigma = torch.sqrt(torch.pow(alpha_sigma, 2).squeeze(-1) + torch.sum(torch.pow(beta_load, 2) * torch.pow(Z_sigma, 2), dim=2))
        return y_mu, y_sigma
    

class FeaturePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeaturePredictor, self).__init__()
        self.key_net = nn.Linear(input_dim, output_dim)
        self.value_net = nn.Linear(input_dim, output_dim)
        self.query = nn.Parameter(torch.randn(output_dim))
        self.alpha = Alpha(output_dim, output_dim, output_dim)


    def forward(self, e):
        key = self.key_net(e)
        value = self.value_net(e)
        attention = F.relu(torch.matmul(key, self.query) / (torch.norm(key, p=2) * torch.norm(value, p=2)))
        attention = attention / torch.sum(attention, dim=0)
        h = torch.sum(attention * value, dim=0)
        y_mu, y_sigma = self.alpha(h)
        return y_mu, y_sigma


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.in_dim = configs.enc_in
        self.out_dim = configs.d_model
        self.feature_extractor = FeatureExtractor(self.in_dim, self.out_dim)
        self.feature_encoder_mu = nn.Linear(1, self.out_dim)
        self.feature_encoder_sigma = nn.Linear(1, self.out_dim)
        self.feature_decoder = FeatureDecoder(self.out_dim, self.out_dim)
        self.feature_predictor = FeaturePredictor(self.out_dim, self.out_dim)

    
    def forward(self, x, y, test=False):
        output, e = self.feature_extractor(x)
        e = e.reshape((e.shape[1], -1, e.shape[2]))
        mu_post = self.feature_encoder_mu(y)
        sigma_post = F.softplus(self.feature_encoder_sigma(y))
        y_mu, y_sigma = self.feature_decoder(e, mu_post, sigma_post)
        z_prior_mu, z_prior_sigma = self.feature_predictor(e)
        
        y_pred_mu, y_pred_sigma = self.feature_decoder(e, z_prior_mu, z_prior_sigma)
        if test:
            return y_pred_mu, y_pred_sigma

        return y_mu, y_sigma, mu_post, sigma_post, z_prior_mu, z_prior_sigma