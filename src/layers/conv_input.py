import torch
import torch.nn as nn

class ConvInput(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super(ConvInput, self).__init__()
        padding = kernel_size // 2  # padding = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out
