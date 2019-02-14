import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def PGConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1, pixelnorm=True, wscale=True, activation='leaky_relu'):
        super(PGConv2d, self).__init()
        # weight initialize
        init = lambda x: nn.init.kaiming_normal(x) if wscale else lambda x: x
        self.conv = nn.Conv2d(ch_in, ch_out, ksize, stride, pad)
        init(self.conv.weight)
        self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2)) if wscale else 1.
        self.conv.weight.data /= self.c
        self.eps = 1e-8

        self.pixelnorm = pixelnorm
        if activation is not None
            self.activation = nn.LeakyReLU(0.2) if activation == 'leaky_relu' else nn.ReLU()
        else:
            self.activation = None
        self.conv.cuda()

    def forward(self, x):
        h = x * self.c
        h = self.conv(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.pixelnorm:
            mean = torch.mean(h * h, 1, keepdim=True == 'leaky_relu' else nn.ReLU()
            h *= torch.rsprt(mean + self.eps)
        return h

class GFirstBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GFirstBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, 4, 1, 3, **layer_settings)
        self.c2 = PGConv2d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, activation=None)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            return self.toRGB(x)
        return x


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, **layer_settings)
        self.c2 = PGConv2d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, activation=None)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            x = self.toRGB(x)
        return x



def Generator(nn.Module):
    def __init__(self, 
        dataset_shape,
        fmap_base           = 4096
        fmap_decay          = 1.0,
        fmap_max            = 512,
        latent_size         = 512,
        normalize_latents   = True,
        wscale          = True,
        pixelnorm       = True,
        leakyrelu       = True):
        super(Generator, self).__init__()

        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]

        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4


        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        if latent_size is None:
            latent_size = nf(0)

        self.normalize_latents = normalize_latents
        layer_settings = {
            'wscale' : wscale,
            'pixelnorm' : pixelnorm,
            'activation' : 'leaky_relu' if leakyrelu else 'relu'
        }
        self.block0 = GFirstBlock(latent_size, nf(1), num_channels, **layer_settings)
        self.blocks = nn.ModuleList([
            GBlock(nf(i-1), nf(i), num_channels, **layer_settings)
            for i in range(2, R)
        ])

        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-8
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)

    def forward(self, x):
        h = x.unsqueeze(2).unsqueeze(3)
        if self.normalize_latents:
            mean = torch.mean(h * h, 1, keepdim=True)
            h *= torch.rsqrt(mean + self.eps)
        h = self.block0(h, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = F.upsample(h, scale_factor=2)
                h = self.blocks[i](h)
            h = F.upsample(h, scale_factor=2)
            ult = self.blocks[self.depth - 1](h, True)
            if self.alpha < 1.0:
                preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.alpha < 1.0 else self.block0.toRGB(h)
            else:
                preult_rgb = 0
            h = preult_rgb * (1 - self.alpha) + ult * self.alpha
        return h


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.c1 = PGConv2d(ch_in, ch_in, **layer_settings)
        self.c2 = PGConv2d(ch_in, ch_out, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class DLastBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DLastBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.stddev = MinibatchStddev()
        self.c1 = PGConv2d(ch_in + 1, ch_in, **layer_settings)
        self.c2 = PGConv2d(ch_in, ch_out, 4, 1, 0, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.stddev(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


def Tstdeps(val):
    return torch.sqrt(((val - val.mean())**2).mean() + 1.0e-8)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.eps = 1.0

    def forward(self, x):
        stddev_mean = Tstdeps(x)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2), x.size(3))
        h = torch.cat((x, new_channel), dim=1)
        return h


class Discriminator(nn.Module):
    def __init__(self,
        dataset_shape, # Overriden based on dataset
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 512,
        wscale          = True,
        pixelnorm       = False,
        leakyrelu       = True):
        super(Discriminator, self).__init__()

        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        self.R = R

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        layer_settings = {
            'wscale': wscale,
            'pixelnorm': pixelnorm,
            'act': 'lrelu' if leakyrelu else 'relu'
        }
        self.blocks = nn.ModuleList([
            DBlock(nf(i), nf(i-1), num_channels, **layer_settings)
            for i in range(R-1, 1, -1)
        ] + [DLastBlock(nf(1), nf(0), num_channels, **layer_settings)])

        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-8
        self.max_depth = len(self.blocks) - 1

    def forward(self, x):
        h = self.blocks[-(self.depth + 1)](x, True)
        if self.depth > 0:
            h = F.avg_pool2d(h, 2)
            if self.alpha < 1.0:
                xlowres = F.avg_pool2d(x, 2)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb

        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = F.avg_pool2d(h, 2)
        h = self.linear(h.squeeze(-1).squeeze(-1))
        return h
