import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, calculate_gain


class wscaled_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, initializer='kaiming', bias=False, lrmul=1.0):
        super(wscaled_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if initializer == 'kaiming':    kaiming_normal_(self.conv.weight, a=calculate_gain('conv2d')/lrmul)
        elif initializer == 'xavier':   xavier_normal(self.conv.weight)
        
        self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0)).cuda()
        else:
            self.bias = 0
        self.conv.weight.data /= (self.c / lrmul)
        self.c = self.c.cuda()

    def forward(self, x):
        h = x * self.c
        h = self.conv(h)
        return h + self.bias.view(1, -1, 1, 1).expand_as(h)

class wscaled_linear(nn.Module):
    def __init__(self, in_channels, out_channels, initializer='kaiming'):
        super(wscaled_liner, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if initializer == 'kaiming':    kaiming_normal_(self.linear.weight, a=calculate_gain('linear')/lrmul)
        elif initializer == 'xavier':   xavier_normal(self.linear.weight)

        self.c = np.sqrt(torch.mean(self.linear.weight.data ** 2))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill(0)).cuda()
        else:
            self.bias = 0
        self.linear.weight.data /= (self.c / lrmul)
        self.c = self.c.cuda()
    
    def forward(self, x):
        h = x * self.c
        h = self.linear(h)
        return h + self.bias.view(1, -1).expand_as(h)


class wscaled_biasLayer(nn.Module):
    def __init__(self, channels, lrmul):
        super(wscaled_biasLayer, self).__init__()
        self.bias = torch.nn.Parameter(torch.FloatTensor(channels).fill_(0)).cuda() * lrmul
    
    def forward(self, x):
        if len(x.shape) == 2:
            h = x + self.bias
        else:
            h = x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return h
