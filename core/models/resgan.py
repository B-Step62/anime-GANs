import torch
from torch import nn
import torch.nn.functional as F
from core.models.modules import ResBlock, OptimizedBlock

class ResGenerator128(nn.Module):

    def __init__(self, z_dim=128, base=1024, norm='batch'):
        super(ResGenerator128, self).__init__()
        self.base = base
        self.bottom_size = 4
        self.z_dim = z_dim

        self.l1 = nn.Linear(z_dim, (self.bottom_size ** 2) * base)

        self.rb1 = ResBlock(base, base//2, upsample=True, norm=norm)
        base = base // 2
        self.rb2 = ResBlock(base, base//2, upsample=True, norm=norm)
        base = base // 2
        self.rb3 = ResBlock(base, base//2, upsample=True, norm=norm)
        base = base // 2
        self.rb4 = ResBlock(base, base//2, upsample=True, norm=norm)
        base = base // 2
        self.rb5 = ResBlock(base, base//2, upsample=True, norm=norm)
        base = base // 2

        last_layers = []
        if norm == 'batch':
            last_layers.append(nn.BatchNorm2d(base))
        last_layers.append(nn.ReLU())
        last_layers.append(nn.Conv2d(base, 3, 3, 1, 1))
        last_layers.append(nn.Tanh())
        self.last = nn.Sequential(*last_layers)

    def forward(self, z):
        bs = z.shape[0]
        h = self.l1(z)
        h = h.view(bs, -1, self.bottom_size, self.bottom_size)
        h = self.rb1(h)
        h = self.rb2(h)
        h = self.rb3(h)
        h = self.rb4(h)
        h = self.rb5(h)
        h = self.last(h)
        return h
            
class ResDiscriminator128(nn.Module):

    def __init__(self, base=32, norm=None, use_sigmoid=True):
        super(ResDiscriminator128, self).__init__()
        self.base = base

        self.ob1 = OptimizedBlock(3, base, downsample=True, norm=norm)
        self.rb2 = ResBlock(base, base*2, downsample=True, norm=norm)
        base *= 2
        self.rb3 = ResBlock(base, base*2, downsample=True, norm=norm)
        base *= 2
        self.rb4 = ResBlock(base, base*2, downsample=True, norm=norm)
        base *= 2
        self.rb5 = ResBlock(base, base*2, downsample=True, norm=norm)
        base *= 2

        linear_layers = []
        linear_layers.append(nn.Linear(base, 1)) 
        if use_sigmoid:
            linear_layers.append(nn.Sigmoid())
        self.last = nn.Sequential(*linear_layers)

    def forward(self, x):
        bs = x.shape[0]
        h = self.ob1(x)
        h = self.rb2(h)
        h = self.rb3(h)
        h = self.rb4(h)
        h = self.rb5(h)
        h = F.avg_pool2d(h, h.shape[3], stride=1)
        h = h.view(bs, self.base*16)
        h = self.last(h)
        return h
