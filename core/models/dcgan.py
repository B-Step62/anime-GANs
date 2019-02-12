import torch
from torch import nn
import torch.nn.functional as F

class Generator64(nn.Module):

    def __init__(self, z_dim=64, top=256, norm='batch'):
        super(Generator64, self).__init__()
        self.top = top
        self.bottom_size = 4
        self.z_dim = z_dim

        linear_layers = []
        linear_layers.append(nn.Linear(z_dim, 1024))
        if norm == 'batch':
            linear_layers.append(nn.BatchNorm1d(1024))
        linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(1024, top*self.bottom_size*self.bottom_size))
        if norm == 'batch':
            linear_layers.append(nn.BatchNorm1d(top*self.bottom_size*self.bottom_size))
        linear_layers.append(nn.ReLU())

        conv_layers1 = []
        conv_layers1.append(nn.ConvTranspose2d(top, top//2, 4, 2, 1))
        if norm == 'batch':
            conv_layers1.append(nn.BatchNorm2d(top//2))
        conv_layers1.append(nn.ReLU())
        top = top // 2
 
        conv_layers2 = []
        conv_layers2.append(nn.ConvTranspose2d(top, top//2, 4, 2, 1))
        if norm == 'batch':
            conv_layers2.append(nn.BatchNorm2d(top//2))
        conv_layers2.append(nn.ReLU())
        top = top // 2
 
        conv_layers3 = []
        conv_layers3.append(nn.ConvTranspose2d(top, top//2, 4, 2, 1))
        if norm == 'batch':
            conv_layers3.append(nn.BatchNorm2d(top//2))
        conv_layers3.append(nn.ReLU())
        top = top // 2
 
        last = []
        last.append(nn.ConvTranspose2d(top, 3, 4, 2, 1))
        last.append(nn.Tanh())

        self.l1 = nn.Sequential(*linear_layers)
        self.c1 = nn.Sequential(*conv_layers1)
        self.c2 = nn.Sequential(*conv_layers2)
        self.c3 = nn.Sequential(*conv_layers3)
        self.last = nn.Sequential(*last)


    def forward(self, z):
        bs = z.shape[0]
        h = self.l1(z)
        h = h.view(bs, self.top, self.bottom_size, self.bottom_size)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.last(h)
        return h
            
class Discriminator64(nn.Module):

    def __init__(self, top=32, norm=None, use_sigmoid=True):
        super(Discriminator64, self).__init__()
        self.top = top
        self.bottom_size = 8

        conv_layers1 = []
        conv_layers1.append(nn.Conv2d(3, top, 4, 2, 1))
        if norm == 'batch':
            conv_layers1.append(nn.BatchNorm2d(top))
        conv_layers1.append(nn.LeakyReLU(0.1))
 
        conv_layers2 = []
        conv_layers2.append(nn.Conv2d(top, top*2, 4, 2, 1))
        if norm == 'batch':
            conv_layers2.append(nn.BatchNorm2d(top*2))
        conv_layers2.append(nn.LeakyReLU(0.1))
        top = top * 2
 
        conv_layers3 = []
        conv_layers3.append(nn.Conv2d(top, top*2, 4, 2, 1))
        if norm == 'batch':
            conv_layers3.append(nn.BatchNorm2d(top*2))
        conv_layers3.append(nn.LeakyReLU(0.1))
        top = top * 2
 
        linear_layers = []
        linear_layers.append(nn.Linear(top * self.bottom_size * self.bottom_size, 512))
        if norm == 'batch':
            linear_layers.append(nn.BatchNorm1d(512))
        linear_layers.append(nn.LeakyReLU(0.2))
        linear_layers.append(nn.Linear(512, 1))
        if use_sigmoid:
            linear_layers.append(nn.Sigmoid())

        self.c1 = nn.Sequential(*conv_layers1)
        self.c2 = nn.Sequential(*conv_layers2)
        self.c3 = nn.Sequential(*conv_layers3)
        self.l1 = nn.Sequential(*linear_layers)

    def forward(self, x):
        bs = x.shape[0]
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = h.view(bs, self.top*4*self.bottom_size*self.bottom_size)
        h = self.l1(h)
        return h


class Generator128(nn.Module):

    def __init__(self, z_dim=64, top=512, norm='batch'):
        super(Generator128, self).__init__()
        self.top = top
        self.bottom_size = 8
        self.z_dim = z_dim

        linear_layers = []
        linear_layers.append(nn.Linear(z_dim, 1024))
        if norm == 'batch':
            linear_layers.append(nn.BatchNorm1d(1024))
        linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(1024, top*self.bottom_size*self.bottom_size))
        if norm == 'batch':
            linear_layers.append(nn.BatchNorm1d(top*self.bottom_size*self.bottom_size))
        linear_layers.append(nn.ReLU())

        conv_layers1 = []
        conv_layers1.append(nn.ConvTranspose2d(top, top//2, 4, 2, 1))
        if norm == 'batch':
            conv_layers1.append(nn.BatchNorm2d(top//2))
        conv_layers1.append(nn.ReLU())
        top = top // 2
 
        conv_layers2 = []
        conv_layers2.append(nn.ConvTranspose2d(top, top//2, 4, 2, 1))
        if norm == 'batch':
            conv_layers2.append(nn.BatchNorm2d(top//2))
        conv_layers2.append(nn.ReLU())
        top = top // 2
 
        conv_layers3 = []
        conv_layers3.append(nn.ConvTranspose2d(top, top//2, 4, 2, 1))
        if norm == 'batch':
            conv_layers3.append(nn.BatchNorm2d(top//2))
        conv_layers3.append(nn.ReLU())
        top = top // 2
 
        last = []
        last.append(nn.ConvTranspose2d(top, 3, 4, 2, 1))
        last.append(nn.Tanh())

        self.l1 = nn.Sequential(*linear_layers)
        self.c1 = nn.Sequential(*conv_layers1)
        self.c2 = nn.Sequential(*conv_layers2)
        self.c3 = nn.Sequential(*conv_layers3)
        self.last = nn.Sequential(*last)


    def forward(self, z):
        bs = z.shape[0]
        h = self.l1(z)
        h = h.view(bs, self.top, self.bottom_size, self.bottom_size)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.last(h)
        return h
            
class Discriminator128(nn.Module):

    def __init__(self, top=64, norm=None, use_sigmoid=True):
        super(Discriminator128, self).__init__()
        self.top = top
        self.bottom_size = 16

        conv_layers1 = []
        conv_layers1.append(nn.Conv2d(3, top, 4, 2, 1))
        if norm == 'batch':
            conv_layers1.append(nn.BatchNorm2d(top))
        conv_layers1.append(nn.LeakyReLU(0.1))
 
        conv_layers2 = []
        conv_layers2.append(nn.Conv2d(top, top*2, 4, 2, 1))
        if norm == 'batch':
            conv_layers2.append(nn.BatchNorm2d(top*2))
        conv_layers2.append(nn.LeakyReLU(0.1))
        top = top * 2
 
        conv_layers3 = []
        conv_layers3.append(nn.Conv2d(top, top*2, 4, 2, 1))
        if norm == 'batch':
            conv_layers3.append(nn.BatchNorm2d(top*2))
        conv_layers3.append(nn.LeakyReLU(0.1))
        top = top * 2
 
        linear_layers = []
        linear_layers.append(nn.Linear(top * self.bottom_size * self.bottom_size, 512))
        if norm == 'batch':
            linear_layers.append(nn.BatchNorm1d(512))
        linear_layers.append(nn.LeakyReLU(0.2))
        linear_layers.append(nn.Linear(512, 1))
        if use_sigmoid:
            linear_layers.append(nn.Sigmoid())

        self.c1 = nn.Sequential(*conv_layers1)
        self.c2 = nn.Sequential(*conv_layers2)
        self.c3 = nn.Sequential(*conv_layers3)
        self.l1 = nn.Sequential(*linear_layers)

    def forward(self, x):
        bs = x.shape[0]
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = h.view(bs, self.top*4*self.bottom_size*self.bottom_size)
        h = self.l1(h)
        return h
