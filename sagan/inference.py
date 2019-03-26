import os
import sys
import glob
import argparse
import shutil
import subprocess

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

sys.path.append(os.pardir)
from models import sagan 
from common.dataset.dataset import FaceDataset
from common.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='MultiClassGAN')
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gen', type=str, required=True)
    parser.add_argument('--N', type=int, default=16)
    parser.add_argument('--row', type=int, default=4)
    parser.add_argument('--mode', choices=['random', 'morphing', 'attention'], default='random')
    args = parser.parse_args()
    return args



def main():
    global device
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


    gen = getattr(sagan, cfg.models.generator.name)(z_dim=cfg.models.generator.z_dim, norm=cfg.models.generator.norm).to(device)

    # restore
    if args.gen is not None:
        if os.path.isfile(args.gen):
            gen.load_state_dict(torch.load(args.gen)['gen_state_dict'])
        else:
            print(f'=> no checkpoint found at {args.gen}')
            sys.exit()

    # arrange path
    top, gen_file = os.path.split(args.gen)
    top, _ = os.path.split(top)
    out_dir = os.path.join(top, 'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    gen_name, ext = os.path.splitext(gen_file)

    for i in range(args.N):
        out_file = os.path.join(out_dir, gen_name + f'_{args.mode}_{i}')
        if args.mode == 'random':
            inference(gen, args, cfg, out_file + '.png')
        elif args.mode == 'morphing':
            #inference(gen, args, cfg, out_file + '.png')
            inference_gif(gen, args, cfg, out_file + '.gif')
        elif args.mode == 'attention':
            inference_attention(gen, args, cfg,out_file + '.gif')

def get_limited_z(size, dim, _min=-0.5, _max=0.5):
    for i in range(size):
        for j in range(dim):
            while(True):
                z_ij = torch.randn((1, 1))
                if torch.max(z_ij) < _max and torch.min(z_ij) > _min:
                    if j == 0:
                        z_i = z_ij
                    else:
                        z_i = torch.cat((z_i, z_ij), dim=1)
                    break
        if i == 0:
            z = z_i
        else:
            z = torch.cat((z, z_i), dim=0)
    return z
            
def inference_attention(gen, args, cfg, out_file, frame_nums=8, fps=20):
    
    out_dir, _ = os.path.split(out_file)
    frame_dir = os.path.join(out_dir, 'frames')
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    gen.train()

    z0 = Variable(get_limited_z(32, cfg.models.generator.z_dim, _min=-1.0, _max=1.0)).to(device)

    image_list = []
    for t in range(frame_nums):
        z1 = Variable(get_limited_z(32, cfg.models.generator.z_dim, _min=-0.5, _max=0.5)).to(device)
        for i in range(fps):
            alpha = i / fps
            z = (1 - alpha) * z0 + alpha * z1
            fake, attn = gen(z)
            attn_size = int(np.sqrt(attn.shape[1]))
            attn = attn.permute(0, 2, 1).view(attn.shape[0], -1, attn_size, attn_size)
            attn = F.upsample(attn, scale_factor=4, mode='bilinear')
            attn0 = attn[:1,3:6,:,:].repeat(1,1,1,1)
            _min = torch.min(attn0)
            _max = torch.max(attn0)
            attn0 = (attn0 - _min) / (_max - _min)
            fake_image = (fake[:1,:,:,:] + 1.0) * 0.5
            fake_image = torch.cat((fake_image, attn0), dim=0)
            save_image(fake_image.data.cpu(), os.path.join(frame_dir, f'{t*fps+i:04d}.png'), nrow=2)
        z0 = z1


    cmd = ['convert','-layers','optimize','-loop','0','-delay','10',f'{frame_dir}/*.png',f'{out_file}']
    subprocess.run(cmd)
    print(f'saving image to {out_file}')
    #image_list[0].save(out_file, save_all=True, append_images=image_list[1:], duration=200, loop=1)


def inference(gen, args, cfg, out_file):

    gen.train()


    if args.mode == 'random':
        z = Variable(get_limited_z(args.row**2, cfg.models.generator.z_dim)).to(device)
    elif args.mode == 'morphing':
        z0 = Variable(get_limited_z(1, cfg.models.generator.z_dim)).to(device)
        z1 = Variable(get_limited_z(1, cfg.models.generator.z_dim)).to(device)
        z2 = Variable(get_limited_z(1, cfg.models.generator.z_dim)).to(device)
        z3 = Variable(get_limited_z(1, cfg.models.generator.z_dim)).to(device)
        for i in range(args.row**2):
            alpha = (i // args.row) / args.row
            beta = (i % args.row) / args.row
            z = z0 if i == 0 else torch.cat((z, (1.0 - alpha) * (1.0 - beta) * z0 + (1.0 - alpha) * beta * z1 + alpha * (1.0 - beta) * z2 + alpha * beta * z3),dim=0)
   
    x_fake, _ = gen(z)

    x_fake = (x_fake[:,:,:,:] + 1.0) * 0.5
    save_image(x_fake.data.cpu(), out_file, nrow=args.row)


def inference_gif(gen, args, cfg, out_file, frame_nums=8, fps=20):

    out_dir, _ = os.path.split(out_file)
    frame_dir = os.path.join(out_dir, 'frames')
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    gen.train()

    z0 = Variable(get_limited_z(32, cfg.models.generator.z_dim, _min=-0.5, _max=0.5)).to(device)

    image_list = []
    for t in range(frame_nums):
        z1 = Variable(get_limited_z(32, cfg.models.generator.z_dim, _min=-0.5, _max=0.5)).to(device)
        for i in range(fps):
            alpha = i / fps
            z = (1 - alpha) * z0 + alpha * z1
            fake, _ = gen(z)
            fake_image = (fake[:9,:,:,:] + 1.0) * 0.5
            save_image(fake_image.data.cpu(), os.path.join(frame_dir, f'{t*fps+i:04d}.png'), nrow=3)
        z0 = z1

    print(f'saving image to {out_file}')

    cmd = ['convert','-layers','optimize','-loop','0','-delay','10',f'{frame_dir}/*.png',f'{out_file}']
    subprocess.run(cmd)
    #image_list[0].save(out_file, save_all=True, append_images=image_list[1:], duration=200, loop=1)
               
    
if __name__ == '__main__':
    main()
