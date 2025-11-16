import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroPad(nn.Module):
    def __init__(self, pad_size=1):
        super(ZeroPad, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        return F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='constant', value=0)
