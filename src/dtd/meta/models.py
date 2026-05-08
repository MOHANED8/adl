from __future__ import annotations

import torch
import torch.nn as nn


class OverfitPredictorLSTM(nn.Module):
    """
    Inputs: (B, T, F)
    Outputs:
      - p_overfit: (B, 1) sigmoid
      - when: (B, 1) predicted relative collapse epoch (regression)
    """

    def __init__(self, in_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_overfit = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head_when = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        y, (h, c) = self.lstm(x)
        last = y[:, -1]
        p = torch.sigmoid(self.head_overfit(last))
        when = self.head_when(last)
        return p, when


class OverfitPredictorTransformer(nn.Module):
    """
    Lightweight temporal transformer baseline.
    """

    def __init__(self, in_features: int, d_model: int = 128, nhead: int = 4, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)
        self.head_overfit = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head_when = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        last = h[:, -1]
        p = torch.sigmoid(self.head_overfit(last))
        when = self.head_when(last)
        return p, when

