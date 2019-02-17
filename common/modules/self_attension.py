import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attension_Layer(nn.Module):
    def __init__(self, in_ch, activation=F.relu, channel_reduce=8):
        super(Attension_Layer, self).__init__()
        self.in_ch = in_ch
        self.activation = activation

        self.query_conv = nn.Conv2d(in_ch, in_ch // channel_reduce, 1, 1, 0)
        self.key_conv = nn.Conv2d(in_ch, in_ch // channel_reduce, 1, 1, 0)
        self.value_conv = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, ch, wi, hi = x.shape
        query = self.query_conv(x).view(bs, -1, wi*hi).permute(0, 2, 1)
        key = self.key_conv(x).view(bs, -1, wi*hi)
        energy = torch.bmm(query, key)
        attension = F.softmax(energy)

        value = self.value_conv(x).view(bs, -1, wi*hi)

        h = torch.bmm(value, attension.permute(0, 2, 1))
        h = h.view(bs, ch, wi, hi)

        h = self.gamma * h + x
        return h, attension
