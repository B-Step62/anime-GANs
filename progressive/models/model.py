# -*- coding: utf-8 -*-
from models.base_model import *
from models.custom_layers import *

def G_conv(in_channels, out_channels, kernel_size, padding, activation, init, param=None, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layers = []
    if use_wscale:
        layers += [wscaled_conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        he_init(layers[-1], init, param)  # init layers
    layers += [activation]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    return layers


def NINLayer(in_channels, out_channels, activation, init, param=None, use_wscale=True):
    layers = []
    if use_wscale:
        layers += [wscaled_conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
    else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
        he_init(layers[-1], init, param)  # init layers
    if not (activation == 'linear'):
        layers += [activation]
    return layers


class Generator(nn.Module):
    def __init__(self, target_size, model_cfg):
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

        R = int(np.log2(target_size))
        assert target_size == 2**R and target_size >= 4
        if self.z_dim is None: 
            z_dim = self.get_nf(0)

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope) if self.activation == 'leaky_relu' else nn.ReLU()
        init_act = 'leaky_relu' if self.activation == 'leaky_relu' else 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        output_init_act = 'tanh' if self.tanh_at_end else 'linear'

        self.pre = None
        self.gblocks = nn.ModuleList()
        self.toRGBs = nn.ModuleList()

        if self.normalize_z:
            self.pre = PixelNormLayer()

        layers = []
        layers += [ReshapeLayer([self.z_dim, 1, 1])]
        layers += G_conv(self.z_dim, self.get_nf(1), 4, 3, act, init_act, negative_slope, self.use_wscale, self.use_batchnorm, self.use_pixelnorm) # instead of linear layer to z 
        layers += G_conv(self.get_nf(1), self.get_nf(1), 3, 1, act, init_act, negative_slope, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)  # first block
        
        self.gblocks.append(nn.Sequential(*layers))

        torgb = NINLayer(self.get_nf(1), 3, output_act, output_init_act, None, self.use_wscale)  # to_rgb layer
        self.toRGBs.append(nn.Sequential(*torgb))

        for I in range(2, R):  # following blocks
            in_ch, out_ch = self.get_nf(I-1), self.get_nf(I)
            layers = [nn.Upsample(scale_factor=2, mode='nearest')]  # upsample
            layers += G_conv(in_ch, out_ch, 3, 1, act, init_act, negative_slope, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            layers += G_conv(out_ch, out_ch, 3, 1, act, init_act, negative_slope, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            self.gblocks.append(nn.Sequential(*layers))
            torgb = NINLayer(out_ch, 3, output_act, output_init_act, None, self.use_wscale)  # to_rgb layer
            self.toRGBs.append(nn.Sequential(*torgb))

    def get_nf(self, stage):
        return min(int(self.z_dim * (4 ** 2) / (2.0 ** (stage * 1.))), 512)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        pre_level, max_level = int(np.floor(cur_level-1)), int(np.ceil(cur_level-1))
        alpha = 1 - (int(cur_level+1) - cur_level)

        h = x
        if self.pre is not None:
            h = self.pre(h)

        for level in range(0, max_level+1):
            h = self.gblocks[level](h)
            if level == pre_level:
                pre_out = self.toRGBs[level](h)
            if level == max_level:
                max_out = self.toRGBs[level](h)
                h = (1.0 - alpha) * resize_activations(pre_out, max_out.size()) + alpha * max_out
        return h

def D_conv(in_channels, out_channels, kernel_size, padding, activation, init, param=None, 
        use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = []
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    if use_wscale:
        layers += [wscaled_conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        he_init(layers[-1], init, param)  # init layers
    layers += [activation]
    if use_layernorm:
        layers += [LayerNormLayer()]  # TODO: requires incoming layer
    return layers


class Discriminator(nn.Module):
    def __init__(self, target_size, model_cfg): 
        super(Discriminator, self).__init__()
        self.target_size = target_size
        self.mbstat_avg = 'all'
        self.mbdisc_kernels = None

        self.model_cfg = model_cfg 
        self.use_wscale = self.model_cfg.use_wscale
        self.use_gdrop = self.model_cfg.use_gdrop
        self.use_layernorm = self.model_cfg.use_layernorm
        self.sigmoid_at_end = self.model_cfg.sigmoid_at_end

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

        self.dblocks = nn.ModuleList()
        self.fromRGBs = nn.ModuleList()
        self.pre = None

        fromrgb = NINLayer(3, self.get_nf(R-1), act, init=init_act, param=negative_slope, use_wscale=self.use_wscale)
        self.fromRGBs.append(nn.Sequential(*fromrgb))

        for I in range(R-1, 1, -1):
            in_ch, out_ch = self.get_nf(I), self.get_nf(I-1)
            layers = D_conv(in_ch, in_ch, 3, 1, act, init_act, negative_slope, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            layers += D_conv(in_ch, out_ch, 3, 1, act, init_act, negative_slope, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            layers += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            self.dblocks.append(nn.Sequential(*layers))
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            fromrgb = NINLayer(3, out_ch, act, init_act, negative_slope, self.use_wscale)
            self.fromRGBs.append(nn.Sequential(*fromrgb))

        layers = []
        in_ch = out_ch = self.get_nf(1)
        if self.mbstat_avg is not None:
            layers += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
            in_ch += 1
        layers += D_conv(in_ch, out_ch, 3, 1, act, init_act, negative_slope, 
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        layers += D_conv(out_ch, self.get_nf(0), 4, 0, act, init_act, negative_slope,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)

        # Increasing Variation Using MINIBATCH Standard Deviation
        if self.mbdisc_kernels:
            layers += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

        out_ch = 1
        # lods.append(NINLayer(net, self.get_nf(0), oc, 'linear', 'linear', None, True, self.use_wscale))
        layers += NINLayer(self.get_nf(0), out_ch, output_act, output_init_act, None, self.use_wscale)
        self.dblocks.append(nn.Sequential(*layers))


    def get_nf(self, stage):
        return min(int(self.model_cfg.initial_f_map / (2.0 ** (stage * 1.0))), 512)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength

        N = len(self.dblocks)
        max_level, pre_level = int(np.floor(N - cur_level)), int(np.ceil(N - cur_level))
        alpha = 1 - (int(cur_level+1) - cur_level)

        h = x
        if self.pre is not None:
            h = self.pre(h)

        if pre_level == max_level:
            h = self.fromRGBs[max_level](h)
            h = self.dblocks[max_level](h)
        else:
            tmp = self.fromRGBs[max_level](h)
            max_in = self.dblocks[max_level](tmp)
            pre_in = self.fromRGBs[pre_level](h)
            h = (1 - alpha) * resize_activations(pre_in, max_in.size()) + alpha * max_in
            h = self.dblocks[pre_level](h)

        for level in range(pre_level+1, N):
            h = self.dblocks[level](h)
        return h

    def get_wscale_mul(self, cur_level):
        modulelists = self.output_layer.get_layers(cur_level)
        wscale_mul = 1.0
        for modulelist in modulelists:
            modules = modulelist._modules
            for key in modules.keys():
                module = modules[key]
                if isinstance(module, WScaleLayer):
                    if hasattr(module, 'scale'):
                        wscale_mul = wscale_mul * module.scale
        return wscale_mul
