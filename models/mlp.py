import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dims=[256, 256], num_classes=10, dropout=0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def mlp_small(num_classes=10, dropout=0.0):
    return MLP(hidden_dims=[256, 256, 256], num_classes=num_classes, dropout=dropout)

def mlp_large(num_classes=10, dropout=0.0):
    return MLP(hidden_dims=[1024, 1024, 1024, 1024, 1024], num_classes=num_classes, dropout=dropout)
