import torch
import torch.nn as nn

class DisplacementNet(nn.Module):
    def __init__(self, num_features=20, hidden_size=512, hidden_layers=8):
        super(DisplacementNet, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_features, hidden_size))
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, 2))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x