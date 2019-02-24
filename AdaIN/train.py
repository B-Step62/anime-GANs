import os
import sys
import glob
import argparse
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

sys.path.append(os.pardir)
from models import vgg
from models.net import Net
from common.dataset.dataset import FaceDataset
from common.dataset.sampler import InfiniteSamplerWrapper
from common.utils.config import Config
from common.utils.poly_lr_scheduler import poly_lr_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vgg', type=str, default='models/vgg_normalized.pth')
    args = parser.parse_args()
    return args


def main():
    global device, cfg
    args = parse_args()
    cfg = Config.from_file(args.config)

    out = cfg.train.out
    if not os.path.exists(out):
        os.makedirs(out)

    # save config and command
    commands = sys.argv
    with open(f'{out}/command.txt', 'w') as f:
        f.write('## Command ################\n\n')
        f.write(f'python {commands[0]} ')
        for command in commands[1:]:
            f.write(command + ' ')
        f.write('\n\n\n')
        f.write('## Args ###################\n\n')
        for name in vars(args):
            f.write(f'{name} = {getattr(args, name)}\n')

    shutil.copy(args.config, f'./{out}')

    # Log 
    logdir = os.path.join(out, 'log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)

    # Set device
    cuda = torch.cuda.is_available()
    if cuda and args.gpu >= 0:
        print('# cuda available! #')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'

    # Set models
    VGG = vgg.VGG
    VGG.load_state_dict(torch.load(args.vgg))
    VGG = torch.nn.Sequential(*list(VGG.children())[:31])
    model = Net(VGG)
    model.to(device)
 
    # Prepare dataset
    content_dataset = FaceDataset(cfg, cfg.train.content_dataset)
    content_loader = torch.utils.data.DataLoader(
            content_dataset,
            batch_size=cfg.train.batchsize,
            shuffle=True,
            num_workers=min(cfg.train.batchsize, 16),
            pin_memory=True,
            drop_last=True)
    style_dataset = FaceDataset(cfg, cfg.train.style_dataset)
    style_loader = torch.utils.data.DataLoader(
            style_dataset,
            batch_size=cfg.train.batchsize,
            sampler = InfiniteSamplerWrapper(style_dataset),
            num_workers=0,
            pin_memory=True,
            drop_last=True)
    style_iter = iter(style_loader)
    print(f'content dataset contains {len(content_dataset)} images.')
    print(f'style dataset contains {len(style_dataset)} images.')

    opt = Adam(model.decoder.parameters(), lr=cfg.train.parameters.lr, betas=(0.5, 0.999))

    iteration = 0
    batchsize = cfg.train.batchsize
    iterations_per_epoch = len(content_loader)
    epochs = cfg.train.iterations // iterations_per_epoch
    for epoch in range(epochs):
        for i, batch in enumerate(content_loader):
            model.train()

            content_images = Variable(batch).to(device)
            style_images = Variable(next(style_iter)).to(device)

            loss_c, loss_s = model(content_images, style_images)
            loss = cfg.train.parameters.lam_c * loss_c + cfg.train.parameters.lam_s * loss_s

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar('loss_content', loss_c.item(), iteration+1)
            writer.add_scalar('loss_style', loss_s.item(), iteration+1)

            lr = poly_lr_scheduler(opt, cfg.train.parameters.lr, iteration, lr_decay_iter=10, max_iter=cfg.train.iterations)
            iteration += 1

            if iteration % cfg.train.print_interval == 0:
                print(f'Epoch:[{epoch}][{iteration}/{cfg.train.iterations}]  loss content:{loss_c.item():.5f} loss style:{loss_s.item():.5f}')

            if iteration % cfg.train.save_interval == 0: 
                if not os.path.exists(os.path.join(out, 'checkpoint')):
                    os.makedirs(os.path.join(out, 'checkpoint'))
                path = os.path.join(out, 'checkpoint', f'iter_{iteration:04d}.pth.tar')
                state = {'state_dict':model.state_dict(),
                         'opt_state_dict':opt.state_dict(),
                         'iteration':iteration,
                        }
                torch.save(state, path)

            if iteration % cfg.train.preview_interval == 0:
                if not os.path.exists(os.path.join(out, 'preview')):
                    os.makedirs(os.path.join(out, 'preview'))
                sample = generate_sample(model, content_images, style_images) 
                save_image(sample.data.cpu(), os.path.join(out, 'preview', f'iter_{iteration:04d}.png'))

def generate_sample(model, content_images, style_images):
    model.eval()
    samples = model.generate(content_images, style_images)
    concat = torch.cat((torch.cat((content_images, style_images), dim=2), samples), dim=2)
    concat = concat[:min(16, content_images.size()[1]),:,:,:]
    concat = (concat + 1.0) * 0.5
    return concat    

if __name__ == '__main__':
    main()
