# -*- coding: utf-8 -*-
import sys, os, time
import argparse
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

sys.path.append(os.pardir)
from common.utils.config import Config
from utils.randomnoisegenerator import RandomNoiseGenerator
from models import model, old_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gen', type=str, required=True)
    parser.add_argument('--gpu', default=0, type=int, help='gpu to use.')
    parser.add_argument('--noise', choices=['random', 'morphing'], default='random')
    parser.add_argument('--row', type=int, default=5)
    parser.add_argument('--N', type=int, default=1)
    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    cfg = Config.from_file(args.config)

    # Dimensionality of the latent vector.
    latent_size = cfg.models.generator.z_dim
    # Use sigmoid activation for the last layer?
    cfg.models.discriminator.sigmoid_at_end = cfg.train.loss_type in ['ls', 'gan']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    # Load model
    assert os.path.exists(args.gen)
    G_state = torch.load(args.gen)
    if 'toRGB.1.0.weight' in G_state.keys():
        G = model.Generator(model_cfg=cfg.models.generator, target_size=cfg.train.target_size).cuda()
    else:
        G = old_model.Generator(model_cfg=cfg.models.generator, target_size=cfg.train.target_size).cuda()
    G.load_state_dict(G_state)
    print(f'load G from {args.gen}')

    # arrange path
    top, gen_file = os.path.split(args.gen)
    top, _ = os.path.split(top)
    out_dir = os.path.join(top, 'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gen_name, ext = os.path.splitext(gen_file)
 
    pattern = gen_name.split('-')
    resol = int(pattern[0].split('x')[0])   
    
    z_generator = RandomNoiseGenerator(cfg.models.generator.z_dim, 'gaussian')

    ## inference ##
    for i in range(args.N):
        out_file = os.path.join(out_dir, gen_name + f'_{args.noise}_{i}')
        if args.noise == 'random':
            inference(G, args, cfg, out_file, resol, z_generator)
        elif args.noise == 'morphing':
            inference_gif(G, args, cfg, out_file, resol, z_generator)

def inference(G, args, cfg, out_file, resol, z_generator):
    # make noise
    if args.noise == 'random':
        z = z_generator(args.row**2)
    elif args.noise == 'morphing':
        
        z0 = z_generator(1)
        z1 = z_generator(1)
        z2 = z_generator(1)
        z3 = z_generator(1)
        for i in range(args.row**2):
            alpha = (i // args.row) / args.row
            beta = (i % args.row) / args.row
            z = z0 if i == 0 else np.concatenate((z, (1.0 - alpha) * (1.0 - beta) * z0 + (1.0 - alpha) * beta * z1 + alpha * (1.0 - beta) * z2 + alpha * beta * z3), axis=0)
    z = Variable(torch.from_numpy(z)).cuda()

    G.eval()
    cur_level = int(np.log2(resol)) - 1
    fake = G(z, cur_level=cur_level)

    print(f'saving image to {out_file}.png')
    save_image((fake.data.cpu() + 1.0) * 0.5, out_file+'.png', nrow=args.row, padding=0)

def inference_gif(G, args, cfg, out_file, resol, z_generator, frame_nums=8, fps=20):
    G.eval()
    cur_level = int(np.log2(resol)) - 1

    # make noise
    z0 = z_generator(4)

    image_list = []
    for t in range(frame_nums):
        z1 = z_generator(4) 
        for i in range(fps):
            alpha = i / fps
            z = (1 - alpha) * z0 + alpha * z1
            z = Variable(torch.from_numpy(z)).cuda()
            fake = G(z, cur_level=cur_level)
            fake_image = torch.cat((torch.cat((fake[0,:,:,:], fake[1,:,:,:]), dim=1), torch.cat((fake[2,:,:,:], fake[3,:,:,:]), dim=1)), dim=2).permute(1,2,0)
            fake_image = np.clip((fake_image.data.cpu().numpy() + 1.0) * 0.5 * 255, 0, 255)
            fake_image = Image.fromarray(fake_image.astype(np.uint8))
            image_list.append(fake_image)
        z0 = z1

    print(f'saving image to {out_file}.gif')
    image_list[0].save(out_file + '.gif', save_all=True, append_images=image_list[1:], duration=120/fps, loop=1)    

if __name__ == '__main__':
    train()
