
# === New models: BiLSTM, BiGRU, and Conv1D+LSTM (drop-in cell) ===
# Assumes:
# - You already ran the cells that create: input_size, seq_len, train_loader, val_loader, test_loader
# - You already have evaluate() and the same early-stopping training loop from your notebook
# - RNNBase is already defined (supports bidirectional + cell={'rnn','lstm','gru'})

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dLSTM(nn.Module):
    def __init__(self, input_size, conv_channels=64, lstm_hidden=96, lstm_layers=1,
                 bidirectional=False, dropout=0.1):
        super().__init__()
        # Conv expects [B, C, T], where C=input_size, T=seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0 if lstm_layers == 1 else dropout
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # (lon, lat)
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)            # [B, F, T]
        x = self.conv(x)                 # [B, C, T]
        x = x.transpose(1, 2)            # [B, T, C]
        out, _ = self.lstm(x)            # [B, T, H*(1 or 2)]
        last = out[:, -1, :]             # [B, H*(1 or 2)]
        return self.head(last)

# Build the new candidate list using the *same logic* as your existing comparison cell
# NOTE: if you already have a 'candidates' list defined earlier, you can extend it.
new_candidates = [
    ("BiLSTM",      lambda: RNNBase(input_size, hidden=96, layers=2, bidirectional=True,  cell='lstm', dropout=0.1)),
    ("BiGRU",       lambda: RNNBase(input_size, hidden=96, layers=2, bidirectional=True,  cell='gru',  dropout=0.1)),
    ("Conv1D+LSTM", lambda: Conv1dLSTM(input_size, conv_channels=64, lstm_hidden=96, lstm_layers=1, bidirectional=False, dropout=0.1)),
]

# If an original `candidates` exists, extend it; otherwise, set it.
try:
    candidates
except NameError:
    candidates = []
candidates = candidates + new_candidates

print("Added models:", [name for name, _ in new_candidates])
