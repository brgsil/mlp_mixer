import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class MLPBlock(nn.Module):

    def __init__(self, block_dim):
        super(MLPBlock, self).__init__()

        self.block_dim = block_dim 

        self.fc1 = nn.Linear(block_dim, block_dim) 
        self.fc2 = nn.Linear(block_dim, block_dim) 

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.fc2(x)

        return y

class Mixer(nn.Module):

    def __init__(self, channels, torch):
        super(Mixer, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.mlp1 = MLPBlock(tokens)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp2 = MLPBlock(tokens)
        
    def forward(self, x):
        y = self.norm1(x)
        # Change y shape: [b, t, c] -> [b, c, t]
        y = torch.transpose(y, 1, 2) 
        y = self.mlp1(y)
        # Change y shape: [b, c, t] -> [b, t, c]
        y = torch.transpose(y, 1, 2) 
        x = x + y
        y = self.norm2(x)
        y = self.mlp2(y)
        y = x + y

        return y

class MLPMixer(nn.Module):

    def __init__(self, channels, tokens):
        super(MLMixer, self).__init__()

        
