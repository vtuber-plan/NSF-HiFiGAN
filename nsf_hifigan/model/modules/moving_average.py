import torch
from torch import nn
from torch.nn import functional as F

from .conv1_keep_length import Conv1dKeepLength

class MovingAverage(Conv1dKeepLength):
    """ Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """
    def __init__(self, feature_dim, window_len, causal=False, \
                 pad_mode='replicate'):
        super(MovingAverage, self).__init__(
            feature_dim, feature_dim, 1, window_len, causal,
            groups=feature_dim, bias=False, tanh=False, \
            pad_mode=pad_mode)
        # set the weighting coefficients
        nn.init.constant_(self.weight, 1/window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, data):
        return super(MovingAverage, self).forward(data)