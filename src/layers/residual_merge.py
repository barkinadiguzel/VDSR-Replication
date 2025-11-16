import torch
import torch.nn as nn

class ResidualMerge(nn.Module):
    def __init__(self):
        super(ResidualMerge, self).__init__()

    def forward(self, ilr, residual):
        return ilr + residual
