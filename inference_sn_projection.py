import os
import sys
import glob
import argparse
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import save_image

from core.models import dcgan, sn_projection 
from core.dataset.dataset import MultiClassFaceDataset
from core.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='MultiClassGAN')
    parser.add_argument('config', type=str)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args



def main():
    global device, cfg
    args = parse_args()
    cfg = Config.from_file(args.config)

    out = cfg.train.out
    if not os.path.exists(out):
        os.makedirs(out)

    # Set device
    cuda = torch.cuda.is_available()
    if cuda and args.gpu >= 0:
        print('# cuda available! #')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'


    # Set model
    if cfg.models.model_type == 'sn_projection':
        gen = getattr(sn_projection, cfg.models.generator.name)(z_dim=cfg.models.generator.z_dim, norm=cfg.models.generator.norm, n_classes=cfg.train.n_classes).to(device)

    checkpoint = torch.load(args.weight)
    iteration = checkpoint['iteration']
    gen_state_dict = checkpoint['gen_state_dict']
    gen.load_state_dict(gen_state_dict)

    # inference

    gen.eval()

    for i in range(4):
        z = Variable(torch.randn((cfg.train.n_classes, cfg.models.generator.z_dim))).to(device)
        x_fake_label = Variable(torch.arange(0, cfg.train.n_classes), dtype=torch.long).to(device)
        x_fake_i = gen(z, y=x_fake_label)
        if i == 0:
            x_fake = x_fake_i
        else:
            x_fake = torch.cat((x_fake, x_fake_i), axis=0)

    if not os.path.exists(os.path.join(out, 'test')):
        os.makedirs(os.path.join(out, 'test'))
    x_fake = (x_fake + 1.0) * 0.5
    save_image(x_fake.data.cpu(), os.path.join(out, 'test', f'iter_{iteration:04d}.png'), nrow=cfg.n_classes)


if __name__ == '__main__':
    main()
