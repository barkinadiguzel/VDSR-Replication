import torch
import torch.nn as nn
from src.layers.conv_first import ConvFirst
from src.layers.conv_block import ConvBlock
from src.layers.conv_last import ConvLast
from src.layers.residual_add import ResidualAdd
from src.layers.pad import ZeroPad2d
from src.config import config

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.padding = ZeroPad2d(pad=config['pad'])  
        self.conv_first = ConvFirst(in_channels=1, out_channels=config['num_features'])
        self.blocks = nn.ModuleList([
            ConvBlock(in_channels=config['num_features'], out_channels=config['num_features'])
            for _ in range(config['num_blocks'])
        ])
        
        self.conv_last = ConvLast(in_channels=config['num_features'], out_channels=1)
        self.residual_merge = ResidualAdd()

    def forward(self, x):
        out = self.padding(x)
        out = self.conv_first(out)
        for block in self.blocks:
            out = self.padding(out)
            out = block(out)
        out = self.padding(out)
        out = self.conv_last(out)
        out = self.residual_merge(x, out)
        return out
