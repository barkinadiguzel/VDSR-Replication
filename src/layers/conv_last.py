import torch
import torch.nn as nn

class ConvLast(nn.Module):
    """
    VDSR'deki son katman: residual tahmini için Conv3x3
    Giriş: 64 kanallı feature map
    Çıkış: 64 kanallı residual map
    """

    def __init__(self, channels=64, kernel_size=3):
        super(ConvLast, self).__init__()
        padding = kernel_size // 2  
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)

    def forward(self, x):
        out = self.conv(x)
        return out
