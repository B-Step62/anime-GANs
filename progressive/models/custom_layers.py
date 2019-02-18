import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, calculate_gain


class wscaled_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, initializer='kaiming', bias=False):
        super(wscaled_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if initializer == 'kaiming':    kaiming_normal_(self.conv.weight, a=calculate_gain('conv2d'))
        elif initializer == 'xavier':   xavier_normal(self.conv.weight)
        
        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))
        self.scale = math.sqrt(2 / (in_channels * kernel_size ** 2))#(torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data/self.scale)

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1).expand_as(x)

