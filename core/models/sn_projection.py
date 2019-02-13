import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from core.models.modules import ResGenBlock, ResDisBlock, OptimizedBlock
from core.models.links import BatchNorm2d

class ResNetGenerator128(torch.nn.Module):
    def __init__(self, base=64, z_dim=128, bottom_width=4, activation=F.relu, norm=None, n_classes=0):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.l1 = nn.Linear(z_dim, (bottom_width ** 2) * base * 16)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        #self.block2 = ResGenBlock(base * 16, base * 16, activation=activation, upsample=True, n_classes=n_classes, norm=norm)
        self.block3 = ResGenBlock(base * 16, base * 8, activation=activation, upsample=True, n_classes=n_classes, norm=norm)
        self.block4 = ResGenBlock(base * 8, base * 4, activation=activation, upsample=True, n_classes=n_classes, norm=norm)
        self.block5 = ResGenBlock(base * 4, base * 2, activation=activation, upsample=True, n_classes=n_classes, norm=norm)
        self.block6 = ResGenBlock(base * 2, base, activation=activation, upsample=True, n_classes=n_classes, norm=norm)
        self.b7 = BatchNorm2d(base)
        self.l7 = nn.Conv2d(base, 3, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l7.weight)
        torch.nn.init.zeros_(self.l7.bias)

    def forward(self, z, y=None):
        if self.n_classes > 0 and y is None:
            print('#!#!#!#!#!#! input y have to be input to conditional Generator. #!#!#!#!#!#!')
            sys.exit()
        h = z
        h = self.l1(h)
        h = h.view(h.shape[0], -1, self.bottom_width, self.bottom_width)
        #h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.block6(h, y)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.l7(h))
        return h

class SNResNetProjectionDiscriminator128(torch.nn.Module):
    def __init__(self, base=64, n_classes=0, activation=F.relu, norm=None):
        super().__init__()
        self.activation = activation
        self.n_classes = n_classes
        self.block1 = OptimizedBlock(3, base, norm=norm)
        self.block2 = ResDisBlock(base, base * 2, activation=activation, downsample=True, norm=norm)
        self.block3 = ResDisBlock(base * 2, base * 4, activation=activation, downsample=True, norm=norm)
        self.block4 = ResDisBlock(base * 4, base * 8, activation=activation, downsample=True, norm=norm)
        self.block5 = ResDisBlock(base * 8, base * 16, activation=activation, downsample=True, norm=norm)
        #self.block6 = ResDisBlock(base * 16, base * 16, activation=activation, downsample=True, norm=norm)
        self.l7 = torch.nn.Linear(base * 16, 1)
        torch.nn.init.xavier_uniform_(self.l7.weight)
        torch.nn.init.zeros_(self.l7.bias)
        if norm == 'spectral':
            spectral_norm(self.l7)

        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes, base * 16)
            torch.nn.init.xavier_uniform_(self.l_y.weight)
            spectral_norm(self.l_y)

    def forward(self, x, y=None):
        if self.n_classes > 0 and y is None:
            print('#!#!#!#!#!#! input y have to be input to conditional Discriminator. #!#!#!#!#!#!')
            sys.exit()
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        #h = self.block6(h)
        h = self.activation(h)
        h = h.sum([2, 3])  #global pooling
        output = self.l7(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output
