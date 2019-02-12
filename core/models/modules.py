import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False, downsample=False, norm='batch'):
        super(ResBlock, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.upsample = upsample
        self.downsample = downsample
        self.norm = norm
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.downsampling = nn.AvgPool2d(2)
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
    def forward_residual_connect(self, x):
        h = self.conv_sc(x)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        elif self.downsample:
            h = self.downsampling(h)
            #out = self.upconv2(out)
        return h
    def forward(self, x):
        h = self.relu(self.norm1(x)) if self.norm=='batch' else self.relu(x)
        h = self.conv1(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        elif self.downsample:
            h = self.downsampling(h)
            #out = self.upconv1(out)
        h = self.relu(self.norm2(h)) if self.norm=='batch' else self.relu(h)
        h = self.conv2(h)
        res = self.forward_residual_connect(x)
        return h + res

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, norm=None):
        super(OptimizedBlock, self).__init__()
        self.norm = norm
        self.downsample = downsample
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)

    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
        if self.norm == 'batch':
            model += nn.BatchNorm2d(out_channels)
        model += [nn.ReLU()]
        model += [nn.Conv2d(out_channels, out_channels, 3, 1, 1)]
        if self.norm == 'batch':
            model += nn.BatchNorm2d(out_channels)
        if self.downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [nn.Conv2d(in_channels, out_channels, 1, 1, 0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)
