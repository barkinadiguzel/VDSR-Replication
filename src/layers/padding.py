import torch
import torch.nn.functional as F

def zero_pad(x, pad=1):
    return F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
