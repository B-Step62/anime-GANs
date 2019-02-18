# -*- coding: utf-8 -*-
from models.base_model import *
from models.custom_layers import *

device = 'cpu'

def G_conv(incoming, in_channels, out_channels, kernel_size, padding, activation, init, param=None, to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True, device='cpu'):
    layers = incoming
    if use_wscale:
        layers += [wscaled_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        he_init(layers[-1], init, param)  # init layers
    layers += [activation]
    #if use_batchnorm:
    #    layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


def NINLayer(incoming, in_channels, out_channels, activation, init, param=None, 
            to_sequential=True, use_wscale=True, device=device):
    layers = incoming
    if use_wscale:
        layers += [wscaled_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
    else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
        he_init(layers[-1], init, param)  # init layers
    if not (activation == 'linear'):
        layers += [activation]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Generator(nn.Module):
    def __init__(self, target_size, model_cfg, xpu):
        super(Generator, self).__init__()
        self.target_size = target_size
        self.model_cfg = model_cfg
        self.z_dim = self.model_cfg.z_dim
        self.normalize_z = self.model_cfg.normalize_z
        self.use_batchnorm = self.model_cfg.use_batchnorm
        self.use_wscale = self.model_cfg.use_wscale
        self.use_pixelnorm = self.model_cfg.use_pixelnorm
        self.activation = self.model_cfg.activation
        self.tanh_at_end = self.model_cfg.tanh_at_end
        self.device = xpu

        R = int(np.log2(target_size))
        assert target_size == 2**R and target_size >= 4
        if self.z_dim is None: 
            z_dim = self.get_nf(0)

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope) if self.activation == 'leaky_relu' else nn.ReLU()
        init_act = 'leaky_relu' if self.activation == 'leaky_relu' else 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        output_init_act = 'tanh' if self.tanh_at_end else 'linear'

        pre = None
        lods = nn.ModuleList()
        nins = nn.ModuleList()
        layers = []

        if self.normalize_z:
            pre = PixelNormLayer()

        layers += [ReshapeLayer([self.z_dim, 1, 1])]
        layers = G_conv(layers, self.z_dim, self.get_nf(1), 4, 3, act, init_act, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm, device=self.device) # instead of linear layer to z 
        net = G_conv(layers, self.get_nf(1), self.get_nf(1), 3, 1, act, init_act, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm, device=self.device)  # first block
    
        lods.append(net)
        nins.append(NINLayer([], self.get_nf(1), 3, output_act, output_init_act, None, True, self.use_wscale, self.device))  # to_rgb layer

        for I in range(2, R):  # following blocks
            in_ch, out_ch = self.get_nf(I-1), self.get_nf(I)
            layers = [nn.Upsample(scale_factor=2, mode='nearest')]  # upsample
            layers = G_conv(layers, in_ch, out_ch, 3, 1, act, init_act, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm, device=self.device)
            net = G_conv(layers, out_ch, out_ch, 3, 1, act, init_act, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm, device=self.device)
            lods.append(net)
            nins.append(NINLayer([], out_ch, 3, output_act, output_init_act, None, True, self.use_wscale, device=self.device))  # to_rgb layer
        self.output_layer = GSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.z_dim * (4 ** 2) / (2.0 ** (stage * 1.))), 512)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(x, y, cur_level, insert_y_at)


def D_conv(incoming, in_channels, out_channels, kernel_size, padding, activation, init, param=None, 
        to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict(), device='cpu'):
    layers = incoming
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    if use_wscale:
        layers += [wscaled_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        he_init(layers[-1], init, param)  # init layers
    layers += [activation]
    if use_layernorm:
        layers += [LayerNormLayer()]  # TODO: requires incoming layer
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Discriminator(nn.Module):
    def __init__(self, target_size, model_cfg, xpu): 
        super(Discriminator, self).__init__()
        self.target_size = target_size
        self.mbstat_avg = 'all'
        self.mbdisc_kernels = None

        self.model_cfg = model_cfg 
        self.use_wscale = self.model_cfg.use_wscale
        self.use_gdrop = self.model_cfg.use_gdrop
        self.use_layernorm = self.model_cfg.use_layernorm
        self.sigmoid_at_end = self.model_cfg.sigmoid_at_end
        self.device = xpu

        R = int(np.log2(target_size))
        assert target_size == 2**R and target_size >= 4
        gdrop_strength = 0.0

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        # input activation
        init_act = 'leaky_relu'
        # output activation
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_init_act = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        nins = nn.ModuleList()
        lods = nn.ModuleList()
        pre = None

        nins.append(NINLayer([], 3, self.get_nf(R-1), act, init=init_act, param=negative_slope, to_sequential=True, use_wscale=self.use_wscale, device=self.device))

        for I in range(R-1, 1, -1):
            in_ch, out_ch = self.get_nf(I), self.get_nf(I-1)
            net = D_conv([], in_ch, in_ch, 3, 1, act, init_act, negative_slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param, device=self.device)
            net = D_conv(net, in_ch, out_ch, 3, 1, act, init_act, negative_slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param, device=self.device)
            net += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            lods.append(nn.Sequential(*net))
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            nin = []
            nin = NINLayer(nin, 3, out_ch, act, init_act, negative_slope, True, self.use_wscale, device=self.device)
            nins.append(nin)

        net = []
        in_ch = out_ch = self.get_nf(1)
        if self.mbstat_avg is not None:
            net += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
            in_ch += 1
        net = D_conv(net, in_ch, out_ch, 3, 1, act, init_act, negative_slope, False, 
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param, device=self.device)
        net = D_conv(net, out_ch, self.get_nf(0), 4, 0, act, init_act, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param, device=self.device)

        # Increasing Variation Using MINIBATCH Standard Deviation
        if self.mbdisc_kernels:
            net += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

        out_ch = 1
        # lods.append(NINLayer(net, self.get_nf(0), oc, 'linear', 'linear', None, True, self.use_wscale))
        lods.append(NINLayer(net, self.get_nf(0), out_ch, output_act, output_init_act, None, True, self.use_wscale, device=self.device))

        self.output_layer = DSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.model_cfg.initial_f_map / (2.0 ** (stage * 1.0))), 512)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)

