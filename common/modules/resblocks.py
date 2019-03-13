import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from common.modules.spectral_norm import SpectralNorm
from common.modules.batchnorm import CategoricalConditionalBatchNorm

class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, stride=1, pad=1, activation=F.relu, upsample=False, n_classes=0, norm=None):
        super(ResGenBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        self.n_classes = n_classes
        self.norm = norm

        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, ksize, stride, pad)
        nn.init.xavier_uniform_(self.conv1.weight, gain=(2**0.5))
        nn.init.zeros_(self.conv1.bias)
        if norm == 'spectral' or norm == 'spectral+batch':
           self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, ksize, stride, pad)
        nn.init.xavier_uniform_(self.conv2.weight, gain=(2**0.5))
        nn.init.zeros_(self.conv2.bias)
        if norm == 'spectral' or norm == 'spectral+batch':
            self.conv2 = SpectralNorm(self.conv2)

        if n_classes > 0 and norm == 'c_batch':
            self.norm1 = CategoricalConditionalBatchNorm(in_channels, n_classes)
            self.norm2 = CategoricalConditionalBatchNorm(hidden_channels, n_classes)
        elif norm == 'batch' or self.norm == 'spectral+batch':
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.conv_sc.weight)
            nn.init.zeros_(self.conv_sc.bias)
            if norm == 'spectral' or norm == 'spectral+batch':
                self.conv_sc = SpectralNorm(self.conv_sc)

    def forward(self, x, y=None):
        assert not (y is None and self.norm=='c_batch') 
        h = x
        if self.norm == 'c_batch':
            h = self.activation(self.norm1(h, y)) 
        elif self.norm == 'batch' or self.norm == 'spectral+batch':
            h = self.activation(self.norm1(h)) 
        else:
            self.activation(h)
        if self.upsample:
            h = F.upsample(h, scale_factor=2)
        h = self.conv1(h)
        if self.norm == 'c_batch':
            h = self.activation(self.norm2(h, y))
        elif self.norm == 'batch' or self.norm == 'spectral+batch':
            h = self.activation(self.norm2(h))
        else:
            self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:
            if self.upsample:
                x = F.upsample(x, scale_factor=2)
            res = self.conv_sc(x)
        else:
            res = x
        return h + res

class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, stride=1, pad=1, activation=F.relu, downsample=False, n_classes=0, norm=None):
        super(ResDisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        self.norm = norm

        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, ksize, stride, pad)
        nn.init.xavier_uniform_(self.conv1.weight, gain=(2**0.5))
        nn.init.zeros_(self.conv1.bias)
        if norm == 'spectral':
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, ksize, stride, pad)
        nn.init.xavier_uniform_(self.conv2.weight, gain=(2**0.5))
        nn.init.zeros_(self.conv2.bias)
        if norm == 'spectral':
            self.conv2 = SpectralNorm(self.conv2)

        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.conv_sc.weight)
            nn.init.zeros_(self.conv_sc.bias)
            if norm == 'spectral':
                self.conv_sc = SpectralNorm(self.conv_sc)

    def forward(self, x):
        use_norm = self.norm != None
        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        if self.learnable_sc:
            res = self.conv_sc(x)
            if self.downsample:
                res = F.avg_pool2d(res, 2)
        else:
            res = x
        return h + res

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1, activation=F.relu, norm=None):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.norm = norm

        self.conv1 = nn.Conv2d(in_channels, out_channels, ksize, stride, pad)
        nn.init.xavier_uniform_(self.conv1.weight, gain=(2**0.5))
        nn.init.zeros_(self.conv1.bias)
        if norm == 'spectral':
            conv1 = self.conv1
            self.conv1 = SpectralNorm(conv1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, ksize, stride, pad)
        nn.init.xavier_uniform_(self.conv2.weight, gain=(2**0.5))
        nn.init.zeros_(self.conv2.bias)
        if norm == 'spectral':
            conv2 = self.conv2
            self.conv2 = SpectralNorm(conv2)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        nn.init.xavier_uniform_(self.conv_sc.weight)
        nn.init.zeros_(self.conv_sc.bias)
        if norm == 'spectral':
            conv_sc = self.conv_sc
            self.conv_sc = SpectralNorm(conv_sc)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        res = self.conv_sc(x)
        res = F.avg_pool2d(res, 2)
        return h + res



