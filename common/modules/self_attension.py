import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from common.modules.spectral_norm import SpectralNorm

class Attension_Layer(nn.Module):
    def __init__(self, in_ch, activation=F.relu, channel_reduce=8, norm=None):
        super(Attension_Layer, self).__init__()
        self.in_ch = in_ch
        self.activation = activation

        self.theta_conv = nn.Conv2d(in_ch, in_ch // channel_reduce, 1, 1, 0)
        self.phi_conv = nn.Conv2d(in_ch, in_ch // channel_reduce, 1, 1, 0)
        self.g_conv = nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0)
        self.last_conv = nn.Conv2d(in_ch//2, in_ch, 1, 1, 0)
        if norm == 'spectral' or 'spectral+batch':
            self.theta_conv = SpectralNorm(self.theta_conv)
            self.phi_conv = SpectralNorm(self.phi_conv)
            self.g_conv = SpectralNorm(self.g_conv)
            self.last_conv = SpectralNorm(self.last_conv)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, ch, wi, hi = x.shape
        location_num = wi * hi
        downsampled_num = location_num // 4

        theta = self.theta_conv(x).view(bs, ch // 8, location_num).permute(0, 2, 1)

        phi = self.phi_conv(x)
        phi = F.max_pool2d(phi, 2)
        phi = phi.view(bs, ch // 8, downsampled_num)

        energy = torch.bmm(theta, phi)
        attn = F.softmax(energy)

        g = self.g_conv(x)
        g = F.max_pool2d(g, 2)
        g = g.view(bs, downsampled_num, ch//2)

        attn_g = torch.bmm(attn, g)
        attn_g = attn_g.view(bs, ch // 2, wi, hi)
        attn_g = self.last_conv(attn_g)

        attn_g = self.gamma * attn_g + x
        return attn_g, attn
