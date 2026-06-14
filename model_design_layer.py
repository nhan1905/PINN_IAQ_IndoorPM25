import torch
import torch.nn as nn
import random

class BiLSTM(nn.Module):
    def __init__(self, past_dim, hidden_dim, seq_len, num_layers=2):
        super(BiLSTM, self).__init__()
        self.bn = nn.BatchNorm1d(past_dim)
        self.encoder = nn.LSTM(
            input_size=past_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x_norm = x.transpose(1, 2)
        x_norm = self.bn(x_norm)
        x_norm = x_norm.transpose(1, 2)
        output, (hidden, cell) = self.encoder(x_norm)
        return output, (hidden, cell)

class PureRegressor(nn.Module):
    def __init__(self, latent_dim, current_dim):
        super(PureRegressor, self).__init__()
        combined_dim = latent_dim + current_dim

        self.net = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, latent, context):
        combined = torch.cat([latent, context], dim=1) 
        return self.net(combined)
        
class MultiHeadRegressor(nn.Module):
    def __init__(self, current_dim, latent_dim):

        super(MultiHeadRegressor, self).__init__()
        combined_dim = current_dim + 1 + latent_dim

        self.shared_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.reg_aer = nn.Linear(64, 1)
        self.reg_p   = nn.Linear(64, 1)
        self.reg_k   = nn.Linear(64, 1)
        self.reg_s   = nn.Linear(64, 1)

    def forward(self, context, prev_c_indoor, latent):
        combined = torch.cat([context, prev_c_indoor, latent], dim=1)
        x = self.shared_net(combined)

        aer = self.reg_aer(x)  # Shape: [Batch, 1]
        p   = self.reg_p(x)    # Shape: [Batch, 1]
        k   = self.reg_k(x)    # Shape: [Batch, 1]
        s   = self.reg_s(x)    # Shape: [Batch, 1]
        
        return torch.cat([aer, p, k, s], dim=1)