

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    """Standard sinusoidal positional encoding 
    (Vaswani et al., 2017, Attention).

    Acts as an orthonormal basis for each position
    """

    def __init__(self,

                  d_model: int,
                    max_len: int = 4096):
        
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        #  (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerHurstRegressor(nn.Module):
    def __init__(self, 
                 input_size: int = 1, 
                 d_model: int = 64, 
                 nhead: int = 4,
                 num_layers: int = 3, 
                 dim_feedforward: int = 256,
                 dropout: float = 0.1, 
                 use_cls_token: bool = True,
                 max_len: int = 4096):
        
        super().__init__()

        self.use_cls_token = use_cls_token


        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)


        # use existing model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)


        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch, T, input_size)

        # very important how PyTorch processes the data and what shape is acceptable

        # This is always confusiong

        h = self.input_proj(x)  # (batch, T, d_model)

        if self.use_cls_token:
            batch_size = h.size(0)
            cls = self.cls_token.expand(batch_size, -1, -1)
            h = torch.cat([cls, h], dim=1) 

        h = self.pos_enc(h)
        h = self.encoder(h)

        if self.use_cls_token:
            pooled = h[:, 0, :]  # CLS token output

        else:
            pooled = h.mean(dim=1)  # global average pooling

        out = self.head(pooled).squeeze(-1)
        return torch.sigmoid(out)  # H in (0, 1)
