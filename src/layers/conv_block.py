import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2  
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out
