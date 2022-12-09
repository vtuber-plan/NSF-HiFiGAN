
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import AvgPool1d

from .discriminator import DiscriminatorP, DiscriminatorS


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs