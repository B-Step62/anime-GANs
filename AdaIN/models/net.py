import torch
import torch.nn as nn
import torch.nn.functional as F

from common.functions.adain import adaptive_instance_normalization as adain
from common.functions.statistics import calc_mean_std

class Net(nn.Module):
    def __init__(self, vgg):
        super(Net, self).__init__()
        vgg = list(vgg.children())
        self.enc_1 = nn.Sequential(*vgg[:4])
        self.enc_2 = nn.Sequential(*vgg[4:11])
        self.enc_3 = nn.Sequential(*vgg[11:18])
        self.enc_4 = nn.Sequential(*vgg[18:31])
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3),
        )

        self.criterion = nn.MSELoss()

        # fix the encoder weight
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, x):
        results = []
        h = x
        for i in range(4):
            h = getattr(self, f'enc_{i+1}')(h)
            results.append(h)
        return results

    def encode(self, x):
        h = x
        for i in range(4):
            h = getattr(self, f'enc_{i+1}')(h)
        return h

    def calc_content_loss(self, x, target):
        assert (x.size() == target.size())
        assert (target.requires_grad is False)
        return self.criterion(x, target)
       
    def calc_style_loss(self, x, target):
        assert (x.size() == target.size())
        assert (target.requires_grad is False)
        x_mean, x_std = calc_mean_std(x)
        target_mean, target_std = calc_mean_std(target)
        loss = self.criterion(x_mean, target_mean) + self.criterion(x_std, target_std)
        return loss

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
  
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s


    def generate(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feat = self.encode(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat
  
        g_t = self.decoder(t)
        return g_t
