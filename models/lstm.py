
# lstm.py

import torch
import torch.nn as nn


class LSTMHurstRegressor(nn.Module):

    def __init__(self, 
                 input_size: int = 1, 
                 hidden_size: int = 128,
                 num_layers: int = 2, 
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1

        feat_dim = hidden_size * direction_factor

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch, T, input_size)
        out, (h_n, c_n) = self.lstm(x)

 
        last = out[:, -1, :]  # (batch, feat_dim)


        h = self.head(last).squeeze(-1)  # (batch,)

        # prediction
        return torch.sigmoid(h)  # H in (0, 1)
    
