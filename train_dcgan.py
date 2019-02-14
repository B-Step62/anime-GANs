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

from dcgan.models import dcgan
from common.dataset.dataset import FaceDataset
from common.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('config', type=str)
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

    loss_type = cfg.train.loss_type

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

    # Set device
    cuda = torch.cuda.is_available()
    if cuda and args.gpu >= 0:
        print('# cuda available! #')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'

    if cfg.models.model_type == 'dcgan':
        gen = getattr(dcgan, cfg.models.generator.name)(z_dim=cfg.models.generator.z_dim, norm=cfg.models.generator.norm).to(device)
        dis = getattr(dcgan, cfg.models.discriminator.name)(norm=cfg.models.discriminator.norm, use_sigmoid=cfg.models.discriminator.use_sigmoid).to(device)

    elif cfg.models.model_type == 'resgan':
        gen = getattr(resgan, cfg.models.generator.name)(z_dim=cfg.models.generator.z_dim, norm=cfg.models.generator.norm).to(device)
        dis = getattr(resgan, cfg.models.discriminator.name)(norm=cfg.models.discriminator.norm, use_sigmoid=cfg.models.discriminator.use_sigmoid).to(device)

    elif cfg.models.model_type == 'sn_projection':
        gen = getattr(sn_projection, cfg.models.generator.name)(z_dim=cfg.models.generator.z_dim, norm=cfg.models.generator.norm, n_classes=cfg.train.n_classes).to(device)
        dis = getattr(sn_projection, cfg.models.discriminator.name)(norm=cfg.models.discriminator.norm, n_classes=cfg.train.n_classes).to(device)


    train_dataset = FaceDataset(cfg, cfg.train.dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batchsize,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True)
    print(f'train dataset contains {len(train_dataset)} images.')

    opt_gen = Adam(gen.parameters(), lr=cfg.train.parameters.g_lr, betas=(0.5, 0.999))
    opt_dis = Adam(dis.parameters(), lr=cfg.train.parameters.d_lr, betas=(0.5, 0.999))

    if loss_type == 'ls':
        criterion = torch.nn.MSELoss().to(device)
    elif loss_type == 'hinge':
        criterion = torch.nn.ReLU().to(device)

    iteration = 0
    batchsize = cfg.train.batchsize
    iterations_per_epoch = len(train_loader)
    epochs = cfg.train.iterations // iterations_per_epoch
    for epoch in range(epochs):
        gen.train()
        dis.train()

        y_real = Variable(torch.ones(batchsize, 1)).to(device)
        y_fake = Variable(torch.zeros(batchsize, 1)).to(device)

        for i, batch in enumerate(train_loader):

            x_real = Variable(batch).to(device)
            z = Variable(torch.randn((batchsize, cfg.models.generator.z_dim))).to(device)

            x_fake = gen(z)

            d_fake = dis(x_fake.detach())
            d_real = dis(x_real)
 
            if loss_type == 'ls':
                d_loss_fake = criterion(d_fake, y_fake)
                d_loss_real = criterion(d_real, y_real)
            elif loss_type == 'wgan-gp':
                d_loss_fake = torch.mean(d_fake)
                d_loss_real = - torch.mean(d_real)
            elif loss_type == 'hinge':
                d_loss_fake = criterion(1.0 + d_fake).mean()
                d_loss_real = criterion(1.0 - d_real).mean()

            d_loss = d_loss_fake + d_loss_real

            if loss_type == 'wgan-gp':
               d_loss_gp = gradient_penalty(x_real, x_fake, dis)
               d_loss += cfg.train.parameters.lambda_gp * d_loss_gp + 0.1 * torch.mean(d_real * d_real)

            opt_gen.zero_grad()
            opt_dis.zero_grad()
            d_loss.backward()
            opt_dis.step()

            z = Variable(torch.randn((batchsize, cfg.models.generator.z_dim))).to(device)
            x_fake = gen(z)
            d_fake = dis(x_fake)
            if loss_type == 'ls':
                g_loss = criterion(d_fake, y_real)
            elif loss_type == 'wgan-gp':
                g_loss = - torch.mean(d_fake)
            elif loss_type == 'hinge':
                g_loss = - torch.mean(d_fake)

            opt_gen.zero_grad()
            opt_dis.zero_grad()
            g_loss.backward()
            opt_gen.step()

            g_lr = poly_lr_scheduler(opt_gen, cfg.train.parameters.g_lr, iteration, lr_decay_iter=10, max_iter=cfg.train.iterations)
            d_lr = poly_lr_scheduler(opt_dis, cfg.train.parameters.d_lr, iteration, lr_decay_iter=10, max_iter=cfg.train.iterations)

            iteration += 1

            if iteration % cfg.train.print_interval == 0:
                if loss_type == 'wgan-gp':
                    print(f'Epoch:[{epoch}][{iteration}/{cfg.train.iterations}]  Loss dis:{d_loss:.5f} dis-gp:{d_loss_gp} gen:{g_loss:.5f}')
                else:
                    print(f'Epoch:[{epoch}][{iteration}/{cfg.train.iterations}]  Loss dis:{d_loss:.5f} gen:{g_loss:.5f}')

            if iteration % cfg.train.save_interval == 0: 
                if not os.path.exists(os.path.join(out, 'checkpoint')):
                    os.makedirs(os.path.join(out, 'checkpoint'))
                path = os.path.join(out, 'checkpoint', f'iter_{iteration:04d}.pth.tar')
                state = {'gen_state_dict':gen.state_dict(),
                         'dis_state_dict':dis.state_dict(),
                         'opt_gen_state_dict':opt_gen.state_dict(),
                         'opt_dis_state_dict':opt_dis.state_dict(),
                         'iteration':iteration,
                        }
                torch.save(state, path)

            if iteration % cfg.train.preview_interval == 0:
                if not os.path.exists(os.path.join(out, 'preview')):
                    os.makedirs(os.path.join(out, 'preview'))
                x_fake = (x_fake[:min(16, batchsize),:,:,:] + 1.0) * 0.5
                save_image(x_fake.data.cpu(), os.path.join(out, 'preview', f'iter_{iteration:04d}.png'))


def gradient_penalty(x_real, x_fake, dis):
    epsilon = torch.rand(x_real.shape[0], 1, 1, 1).to(device).expand_as(x_real)
    x_hat = Variable(epsilon * x_real.data + (1 - epsilon) * x_fake.data, requires_grad=True)
    d_hat = dis(x_hat)

    grad = torch.autograd.grad(outputs=d_hat,
                               inputs=x_hat,
                               grad_outputs=torch.ones(d_hat.shape).to(device),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = torch.mean((grad_norm - 1) ** 2)

    return d_loss_gp

def poly_lr_scheduler(optimizer, init_lr, iteration, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteration % lr_decay_iter or iteration > max_iter:
        return optimizer

    lr = init_lr*(1 - iteration/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
