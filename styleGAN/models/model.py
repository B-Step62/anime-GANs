# -*- coding: utf-8 -*-
from models.base_model import *
from models.custom_layers import *


class G_Conv_Layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, init, param=None, use_wscale=True, apply_bias=True, use_blur=True, use_noise=False, use_pixelnorm=False, use_instancenorm=True, lrmul=1.0):
        super(G_Conv_Layer, self).__init__():
        if use_wscale:
            self.conv = wscaled_conv2d(in_channels=in_channels,  out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, lrmul=lrmul)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            he_init(self.conv, init, param)  # init layers
        self.activation = activation
        
    def forward(self, x, noise=None):
        h = self.conv(x)
        if use_noise:
            h = apply_noise(h, noise, randamize=False)
        if apply_bias:
            h = apply_bias(h)
        if self.activation is not None:
            h = self.activation(h)
        if use_pixel_norm:
            h = pixel_norm(h)
        if use_instancenorm:
            h = instance_norm(h)
        return h
        

class NINLayer(nn.Module):

    def __init__(self, in_channels, out_channels, activation, init, param=None, use_wscale=True, apply_bias=True, lrmul=1.0):
        super(NINLayer, self).__init__():
        if use_wscale:
            self.conv = wscaled_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, lrmul=1.0)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            he_init(layers[-1], init, param)  # init layers
        self.activation = activation

    def forward(self, x):
        h = self.conv(x)
        if apply_bias:
            h = apply_bias(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation, init, param, use_wscale=True, use_noise=True, use_pixelnorm=False, use_instancenorm=True, upsample=False, adain=True):
        self.conv1 = G_Conv_Layer(in_channels, out_channels, 3, 1, activation, init, param, 
                                  use_wscale=use_wscale,           
                                  apply_bias=True,
                                  use_blur=True,
                                  use_noise=use_noise,
                                  use_pixelnorm=use_pixelnorm,
                                  use_instancenorm=use_instancenorm):

        self.conv2 = G_Conv_Layer(out_channels, out_channels, 3, 1, activation, init, param,        
                                  use_wscale=use_wscale,           
                                  apply_bias=True,
                                  use_blur=False,
                                  use_noise=use_noise,
                                  use_pixelnorm=use_pixelnorm,
                                  use_instancenorm=use_instancenorm):

        self.upsample = upsample

    def forward(x, style=None):
        h = x
        if self.upsample:
            h = F.upsample(scale_factor=2, mode='nearest')
        h = self.conv1(h)
        if adain:
            assert style is not None
            h = adain(h, style)
        h = self.conv2(h)
        return h
        


class G_synthesis(nn.Module):

    def __init__(self, target_size, model_cfg):
        super(G_synthesis, self).__init__()
        self.target_size = target_size
        self.model_cfg = model_cfg
        self.dlatents_size = self.model_cfg.dlatents_size # latents W  paper : 512

        self.z_dim = self.model_cfg.z_dim
        self.use_wscale = self.model_cfg.use_wscale
        self.use_noise = self.model_cfg.use_noise
        self.use_pixelnorm = self.model_cfg.use_pixelnorm
        self.use_instance_norm = self.model_cfg.use_instance_norm
        self.activation = self.model_cfg.activation
        self.blur_filter = [1, 2, 1]
        self.tanh_at_end = self.model_cfg.tanh_at_end
        

        R = int(np.log2(target_size))
        assert target_size == 2**R and target_size >= 4
        if self.z_dim is None: 
            z_dim = self.get_nf(0)


        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope) if self.activation == 'leaky_relu' else nn.ReLU()
        init_act = 'leaky_relu' if self.activation == 'leaky_relu' else 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else None
        output_init_act = 'tanh' if self.tanh_at_end else 'linear'

        self.pre = None
        self.gblocks = nn.ModuleList()
        self.toRGBs = nn.ModuleList()

        ### TODO
        # Constant input layer


        # First Block
        self.gblocks.append(GBlock(self.z_dim, self.get_nf(1), 4, 3, act, init_act, negative_slope,
                                   use_wscale = self.use_wscale,
                                   use_noise = self.use_noise,
                                   use_pixelnorm = use_pixelnorm,
                                   use_instancenorm = use_instancenorm,
                                   upsample = False,
                                   adain=True) 
        torgb = NINLayer(self.get_nf(1), 3, output_act, output_init_act, None, use_wscale=self.use_wscale, apply_bias=True) 
        self.toRGBs.append(nn.Sequential(*torgb))


        # Following Blocks
        for I in range(2, R): 
            in_ch, out_ch = self.get_nf(I-1), self.get_nf(I)
            self.gblocks.append(GBlock(in_ch, out_ch, 3, 1, act, init_act, negative_slope,
                                       use_wscale = self.use_wscale
                                       use_noise = self.use_noise,
                                       use_pixelnorm = use_pixelnorm,
                                       use_instancenorm = use_instancenorm,
                                       upsample = True,
                                       adain=True) 
            torgb = NINLayer(out_ch, 3, output_act, output_init_act, None, use_wscale=self.use_wscale, apply_bias=True)
            self.toRGBs.append(nn.Sequential(*torgb))

    def get_nf(self, stage):
        return min(int(self.z_dim * (4 ** 2) / (2.0 ** (stage * 1.))), 512)

    def forward(self, z, style=None, cur_level=None):
        pre_level, max_level = int(np.floor(cur_level-1)), int(np.ceil(cur_level-1))
        alpha = 1 - (int(cur_level+1) - cur_level)

        h = z
        if self.pre is not None:
            h = self.pre(h)

        for level in range(0, max_level+1):
            h = self.gblocks[level](h, style)
            if level == pre_level:
                pre_out = self.toRGBs[level](h)
            if level == max_level:
                max_out = self.toRGBs[level](h)
                h = (1.0 - alpha) * resize_activations(pre_out, max_out.size()) + alpha * max_out
        return h
                  

def mapping_Linear(channels, activation, init, param=None, use_wscale=False, lrmul=1.0):
    layers = []
    if use_wscale:
        layers += [wscaled_linear(in_channels=channels, out_channels=channels, lrmul=lrmul)]
    else:
        layers += [nn.Linear(in_channels=channels, out_channels=channels)]
        he_init(layers[-1], init, param)  # init layers
    layers += [activation]
    return layers


class G_mapping(nn.Module):

    def __init__(self, model_cfg, xpu):
        super(G_mapping, self).__init__()
        self.latent_size = model_cfg.latent_size #latent(z) size   paper : 512
        self.dlatent_size = model_cfg.dlatent_size #latent(W) size  paper : 512
        self.mapping_layers = model_cfg.mapping_layers # paper : 8
        self.mapping_fmaps = model_cfg.mapping_fmaps  # paper : 512
        self.mapping_lrmul = model_cfg.mapping_lrmul # Learning rate multiplier for mapping layers  paper : 0.01
        self.activation = model_cfg.activation # paper : leaky_relu
        self.use_wscale = model.cfg.use_wscale
        self.normalize_latents = model.normalize_latents

        act = nn.LeakyReLU(negative_slope = 0.2) if self.activation == 'leaky_relu' else nn.ReLU()

        layers = []
        for i in range(self.mapping_layers):
            layers += mapping_Linear(self.mapping_fmaps, act, self.use_wscale, self.mapping_lrmul) 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class D_Conv_Layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, init, param=None, gdrop_param=dict(), use_gdrop=True, use_wscale=True, apply_bias=True, use_blur=True, lrmul=1.0):
        super(D_Conv_Layer, self).__init__():
        self.g_drop = GDropLayer(**gdrop_param)
        if use_wscale:
            self.conv = wscaled_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, lrmul=lrmul)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            he_init(self.conv, init, param)  # init layers
        
    def forward(self, x):
        h = x
        h = self.g_drop(h)
        h = self.conv(h)
        if apply_bias:
            h = apply_bias(h)
        if activation is not None:
            h = activation(h)
        return h

class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation, init, param, use_gdrop, gdrop_param=dict(), use_wscale=True, downsample=False):
        self.conv1 = D_Conv_Layer(in_channels, out_channels, 3, 1, activation, init, param, gdrop_param,
                                  use_gdrop = use_gdrop, 
                                  use_wscale = use_wscale,           
                                  apply_bias = True,
                                  use_blur = True)

        self.conv2 = D_Conv_Layer(out_channels, out_channels, 3, 1, activation, init, param, gdrop_param,
                                  use_gdrop = use_gdrop,        
                                  use_wscale = use_wscale,           
                                  apply_bias = True,
                                  use_blur = False):

        self.downsample = downsample

    def forward(x, style=None):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        if self.downsample:
        h = F.avgpool_2d(h, 2) 
        return h
        

        

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

        self.fromRGBs = nn.ModuleList()
        self.dblocks = nn.ModuleList()


        # FromRGB Layer of First Block
        fromrgb = NINLayer(3, self.get_nf(R-1), act, init=init_act, param=negative_slope, use_wscale=self.use_wscale)
        self.fromRGBs.append(nn.Sequential(*fromrgb))

        for I in range(R-1, 1, -1):
            in_ch, out_ch = self.get_nf(I), self.get_nf(I-1)
            self.dblocks.append(DBlock(in_ch, in_ch, 3, 1, act, init_act, negative_slope, gdrop_param,
                                       use_gdrop = self.use_gdrop,
                                       use_wscale = self.use_wscale
                                       downsample = True)
            fromrgb = NINLayer(3, out_ch, act, init_act, negative_slope, self.use_wscale)
            self.fromRGBs.append(nn.Sequential(*fromrgb))

        # Last Block with Minibatch sdv
        layers = []
        in_ch = out_ch = self.get_nf(1)
        if self.mbstat_avg is not None:
            layers += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)
            in_ch += 1
        layers += [D_conv(in_ch, out_ch, 3, 1, act, init_act, negative_slope, 
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)]
        layers += [D_conv(out_ch, self.get_nf(0), 4, 0, act, init_act, negative_slope,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)]

        # Increasing Variation Using MINIBATCH Standard Deviation
        if self.mbdisc_kernels:
            layers += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

        out_ch = 1
        layers += NINLayer(self.get_nf(0), out_ch, output_act, output_init_act, None, self.use_wscale)
        dblocks.append(nn.Sequential(*net))

    def get_nf(self, stage):
        return min(int(self.model_cfg.initial_f_map / (2.0 ** (stage * 1.0))), 512)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength

        N = len(dblocks)
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
            h = self.gblocks[level](h)
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
