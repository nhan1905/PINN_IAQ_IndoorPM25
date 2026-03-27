import torch
import torch.nn as nn

# Define Model Design Layer
# LSTM - Standard unidirectional LSTM for sequence-to-vector prediction
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(LSTM, self).__init__()
        self.seq_len = seq_len

        # LSTM layer (encoder)
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=2, batch_first=True, bidirectional=False)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Returns latent representation: (batch, hidden_dim)
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1, :, :]  # last layer hidden state
        return latent

# CNN-LSTM - Conv1D extracts local features, LSTM captures temporal dependency
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, cnn_channels=64, kernel_size=3):
        super(CNN_LSTM, self).__init__()
        self.seq_len = seq_len

        # Conv1d: (batch, input_dim, seq_len) -> (batch, cnn_channels, seq_len')
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        # LSTM processes conv output: (batch, seq_len, cnn_channels)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, bidirectional=False)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv(x)       # (batch, cnn_channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_channels)
        _, (hidden, _) = self.lstm(x)
        latent = hidden[-1, :, :]  # (batch, hidden_dim)
        return latent

# BiLSTM - Standard Bidirectional LSTM for sequence-to-vector prediction
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(BiLSTM, self).__init__()
        self.seq_len = seq_len
        
        # BiLSTM layer (encoder - kept for compatibility with utils.py)
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                               num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Returns latent representation: (batch, hidden_dim * 2)
        _, (hidden, _) = self.encoder(x)
        latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return latent
