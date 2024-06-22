import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, Nfeat, Nhidden):
        super().__init__()
        self.flatten = nn.Flatten()
        self.NN_layers = nn.Sequential(
            nn.Linear(Nfeat, Nhidden),
            nn.Linear(Nhidden, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.NN_layers(x)
        return out
