import torch
import torch.nn as nn

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
    def __init__(
        self,
        current_dim,
        latent_dim,
        context_mean,
        context_std,
        indoor_mean,
        indoor_std
    ):
        super().__init__()

        # Current context + previous indoor PM2.5 + two past-PM2.5
        # slope features + the BiLSTM latent representation.
        combined_dim = current_dim + 1 + 2 + latent_dim

        self.register_buffer(
            "context_mean",
            torch.as_tensor(context_mean, dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "context_std",
            torch.as_tensor(context_std, dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "indoor_mean",
            torch.as_tensor(indoor_mean, dtype=torch.float32).view(1, 1)
        )
        self.register_buffer(
            "indoor_std",
            torch.as_tensor(indoor_std, dtype=torch.float32).view(1, 1)
        )

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
        self.reg_p = nn.Linear(64, 1)
        self.reg_k = nn.Linear(64, 1)
        self.reg_s = nn.Linear(64, 1)

    def forward(
        self,
        context,
        prev_c_indoor,
        slope_1,
        slope_5,
        latent
    ):
        context_normalized = (
            context - self.context_mean
        ) / self.context_std

        indoor_normalized = (
            prev_c_indoor - self.indoor_mean
        ) / self.indoor_std

        # Both slopes have units of indoor PM2.5 change per time step.
        # Their expected mean is approximately zero, so only scale them.
        slope_1_normalized = slope_1 / self.indoor_std
        slope_5_normalized = slope_5 / self.indoor_std

        combined = torch.cat(
            [
                context_normalized,
                indoor_normalized,
                slope_1_normalized,
                slope_5_normalized,
                latent
            ],
            dim=1
        )

        x = self.shared_net(combined)

        aer = self.reg_aer(x)
        p = self.reg_p(x)
        k = self.reg_k(x)
        s = self.reg_s(x)

        return torch.cat([aer, p, k, s], dim=1)
