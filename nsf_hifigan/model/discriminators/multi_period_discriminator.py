
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from .discriminator import DiscriminatorP, DiscriminatorS

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods: List[int]=[2, 3, 5, 7, 11, 17, 23, 37], use_spectral_norm: bool=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, g=None):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

