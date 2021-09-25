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

    def __init__(self, channels, tokens):
        super(Mixer, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.mlp1 = MLPBlock(tokens)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp2 = MLPBlock(channels)
        
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

    def __init__(self, image_size, in_channels, patch_size, channels,  num_mlp):
        super(MLPMixer, self).__init__()

        self.l = int(image_size / patch_size)
        self.image_size = image_size
        self.channels = channels
        self.in_channels = in_channels
        self.tokens = int(image_size*image_size / (patch_size * patch_size))
        self.embed = nn.Conv2d(in_channels, channels, patch_size, stride=patch_size)

        self.mlp_layers = nn.ModuleList([
                Mixer(channels, self.tokens) for _ in range(num_mlp)
            ])
        
        self.desembed = nn.Conv2d(channels, in_channels * patch_size * patch_size, 1)  
        

    def forward(self, x):
        batch = x.shape[0]

        y = self.embed(x)

        y = y.reshape(batch, self.channels, self.tokens)
        y = y.transpose(1,2)
        k=1
        for mixer in self.mlp_layers:
            print(f'Layer{k}')
            y = mixer(y)
            k+=1

        y = y.transpose(1,2)
        y = y.reshape(batch, self.channels, self.l, self.l)

        y = self.desembed(y)

        y = y.reshape(batch, y.shape[1], self.tokens).transpose(1,2).reshape(batch, self.in_channels, self.image_size, self.image_size)

        return y





