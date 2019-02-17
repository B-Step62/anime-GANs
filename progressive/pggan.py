# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
from scipy.misc import imsave

import torch
import torch.optim as optim
from torch.autograd import Variable
from common.functions.gradient_penalty import gradient_penalty
from utils.logger import Logger

class PGGAN():
    def __init__(self, G, D, dataset, z_generator, gpu, cfg):
        self.G = G
        self.D = D
        self.dataset = dataset
        self.cfg = cfg
        self.z_generator = z_generator
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.logger = Logger('./logs/' + self.current_time + "/")
        self.use_cuda = gpu >= 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

        self.bs_map = {2**R: self.get_bs(2**R) for R in range(2, 11)} # batch size map keyed by resolution_level
        self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}

        self.restore_model()

    def restore_model(self):
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.time = self.current_time
        self._from_resol = 4
        self._phase = 'stabilize'
        self._epoch = 0
        self.is_restored = False
        self.sample_dir = os.path.join(self.cfg.train.out, 'samples')
        self.checkpoint_dir = os.path.join(self.cfg.train.out, 'checkpoint')
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        return 

    def get_bs(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)

    def register_on_gpu(self):
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def create_optimizer(self):
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.cfg.train.parameters.g_lr, betas=(self.cfg.train.parameters.beta1, self.cfg.train.parameters.beta2))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.cfg.train.parameters.d_lr, betas=(self.cfg.train.parameters.beta1, self.cfg.train.parameters.beta2))
        
    def create_criterion(self):
        # w is for gan
        if self.cfg.train.loss_type== 'ls':
            self.adv_criterion = lambda p,t,w: torch.mean((p-t)**2)  # sigmoid is applied here
        elif self.cfg.train.loss_type == 'wgan-gp':
            self.adv_criterion = lambda p,t,w: (-2*t+1) * torch.mean(p)
        elif self.cfg.train.loss_type == 'gan':
            self.adv_criterion = lambda p,t,w: -w*(torch.mean(t*torch.log(p+1e-8)) + torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % self.cfg.train.loss_type)

    def compute_adv_loss(self, prediction, target, w):
        return self.adv_criterion(prediction, target, w)

    def compute_additional_g_loss(self):
        return 0.0

    def compute_additional_d_loss(self, x_real, x_fake):
        # drifting loss and gradient penalty, weighting inside this function
        if self.cfg.train.loss_type == 'wgan-gp':
            d_loss_gp = gradient_penalty(x_real, x_fake, self.D)
            d_loss_drift = 0. # TODO
            return d_loss_drift + d_loss_gp * self.cfg.train.parameters.lambda_gp
        else:
            return 0.

    def _get_data(self, d):
        return d.data[0] if isinstance(d, Variable) else d

    def compute_G_loss(self):
        g_adv_loss = self.compute_adv_loss(self.d_fake, True, 1)
        g_add_loss = self.compute_additional_g_loss()
        self.g_adv_loss = self._get_data(g_adv_loss)
        self.g_add_loss = self._get_data(g_add_loss)
        return g_adv_loss + g_add_loss

    def compute_D_loss(self):
        self.d_adv_loss_real = self.compute_adv_loss(self.d_real, True, 0.5)
        self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake, False, 0.5) * self.cfg.train.parameters.lambda_d_fake
        d_adv_loss = self.d_adv_loss_real + self.d_adv_loss_fake
        d_add_loss = self.compute_additional_d_loss(self.real, self.fake)
        self.d_adv_loss = self._get_data(d_adv_loss)
        self.d_add_loss = self._get_data(d_add_loss)
 
        return d_adv_loss + d_add_loss

    def _rampup(self, epoch, rampup_length):
        if epoch < rampup_length:
            p = max(0.0, float(epoch)) / float(rampup_length)
            p = 1.0 - p
            return np.exp(-p*p*5.0)
        else:
            return 1.0

    def _rampdown_linear(self, epoch, num_epochs, rampdown_length):
        if epoch >= num_epochs - rampdown_length:
            return float(num_epochs - epoch) / rampdown_length
        else:
            return 1.0

    '''Update Learning rate
    '''
    def update_lr(self, cur_nimg):
        for param_group in self.optim_G.param_groups:
            lrate_coef = self._rampup(cur_nimg / 1000.0, self.cfg.train.rampup_kimg)
            lrate_coef *= self._rampdown_linear(cur_nimg / 1000.0, self.cfg.train.total_kimg, self.cfg.train.rampdown_kimg)
            param_group['lr'] = lrate_coef * self.cfg.train.parameters.g_lr
        for param_group in self.optim_D.param_groups:
            lrate_coef = self._rampup(cur_nimg / 1000.0, self.cfg.train.rampup_kimg)
            lrate_coef *= self._rampdown_linear(cur_nimg / 1000.0, self.cfg.train.total_kimg, self.cfg.train.rampdown_kimg)
            param_group['lr'] = lrate_coef * self.cfg.train.parameters.d_lr

    def postprocess(self):
        # TODO: weight cliping or others
        pass

    def _numpy2var(self, x):
        var = Variable(torch.from_numpy(x))
        if self.use_cuda:
            var = var.cuda()
        return var

    def _var2numpy(self, var):
        if self.use_cuda:
            return var.cpu().data.numpy()
        return var.data.numpy()

    def compute_noise_strength(self):
        if not self.cfg.models.discriminator.add_noise:
            return 0

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + np.clip(torch.mean(self.d_real).data[0], 0.0, 1.0) * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        return strength

    def preprocess(self, z, real):
        self.z = self._numpy2var(z)
        self.real = Variable(real).cuda()
        #self.real = self._numpy2var(real)

    def forward_G(self, cur_level):
        self.d_fake = self.D(self.fake, cur_level=cur_level)
    
    def forward_D(self, cur_level, detach=True):
        self.fake = self.G(self.z, cur_level=cur_level)
        strength = self.compute_noise_strength()
        self.d_real = self.D(self.real, cur_level=cur_level, gdrop_strength=strength)
        self.d_fake = self.D(self.fake.detach() if detach else self.fake, cur_level=cur_level)

    def backward_G(self):
        g_loss = self.compute_G_loss()
        g_loss.backward()
        self.optim_G.step()
        self.g_loss = self._get_data(g_loss)

    def backward_D(self, retain_graph=False):
        d_loss = self.compute_D_loss()
        d_loss.backward(retain_graph=retain_graph)
        self.optim_D.step()
        self.d_loss = self._get_data(d_loss)

    def report(self, it, num_it, phase, resol):
        formation = 'Iter[%d|%d], %s, %s, G: %.3f, D: %.3f, G_adv: %.3f, G_add: %.3f, D_adv: %.3f, D_add: %.3f'
        values = (it, num_it, phase, resol, self.g_loss, self.d_loss, self.g_adv_loss, self.g_add_loss, self.d_adv_loss, self.d_add_loss)
        print(formation % values)

    def tensorboard(self, it, num_it, phase, resol, samples):
        # (1) Log the scalar values
        prefix = str(resol)+'/'+phase+'/'
        info = {prefix + 'G_loss': self.g_loss,
                prefix + 'G_adv_loss': self.g_adv_loss,
                prefix + 'G_add_loss': self.g_add_loss,
                prefix + 'D_loss': self.d_loss,
                prefix + 'D_adv_loss': self.d_adv_loss,
                prefix + 'D_add_loss': self.d_add_loss,
                prefix + 'D_adv_loss_fake': self._get_data(self.d_adv_loss_fake),
                prefix + 'D_adv_loss_real': self._get_data(self.d_adv_loss_real)}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, it)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.G.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('G/' + prefix +tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('G/' + prefix +tag + '/grad', self._var2numpy(value.grad), it)

        for tag, value in self.D.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('D/' + prefix + tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('D/' + prefix + tag + '/grad',
                                          self._var2numpy(value.grad), it)

        # (3) Log the images
        # info = {'images': samples[:10]}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, it)

    def train_phase(self, R, phase, batch_size, cur_nimg, from_it, total_it):
        assert total_it >= from_it
        resol = 2 ** (R+1)

        self.dataset.shuffle()
        dataset_len = len(self.dataset)

        for it in range(from_it, total_it):
            if phase == 'stabilize':
                cur_level = R
            else:
                cur_level = R + total_it/float(from_it)
            cur_resol = 2 ** int(np.ceil(cur_level+1))

            # set current image size
            self.dataset.setsize([cur_resol, cur_resol])

            # get a batch noise and real images
            z = self.z_generator(batch_size)

            for b in range(batch_size):
                if b == 0:
                    one = self.dataset[it * batch_size % dataset_len]
                    x = one.view(1, -1, one.shape[1], one.shape[2])
                else:
                    x = torch.cat((x, self.dataset[(it * batch_size + b) % dataset_len].view(1, x.shape[1], x.shape[2], x.shape[3])), dim=0)

            # ===preprocess===
            self.preprocess(z, x)
            self.update_lr(cur_nimg)

            # ===update D===
            self.optim_G.zero_grad()
            self.optim_D.zero_grad()
            self.forward_D(cur_level, detach=True)
            self.backward_D()

            # ===update G===
            self.optim_G.zero_grad()
            self.optim_D.zero_grad()
            self.forward_G(cur_level)
            self.backward_G()

            # ===report ===
            if it % self.cfg.train.print_interval == 0:
                self.report(it, total_it, phase, cur_resol)

            cur_nimg += batch_size

            # ===generate sample images===
            samples = []
            if (it % self.cfg.train.preview_interval == 0) or it == total_it-1:
                samples = self.sample()
                imsave(os.path.join(self.sample_dir,
                                    '%dx%d-%s-%s.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))), samples)

            # ===tensorboard visualization===
            if (it % self.cfg.train.preview_interval == 0) or it == total_it - 1:
                self.tensorboard(it, total_it, phase, cur_resol, samples)

            # ===save model===
            if (it % self.cfg.train.save_interval == 0 and it > 0) or it == total_it-1:
                self.save(os.path.join(self.checkpoint_dir, '%dx%d-%s-%s' % (cur_resol, cur_resol, phase, str(it).zfill(6))))
        
    def train(self):
        # prepare
        self.create_optimizer()
        self.create_criterion()
        self.register_on_gpu()

        to_level = int(np.log2(self.cfg.train.target_size))
        from_level = int(np.log2(self._from_resol))
        assert 2**to_level == self.cfg.train.target_size and 2**from_level == self._from_resol and to_level >= from_level >= 2

        train_kimg = int(self.cfg.train.stabilizing_kimg * 1000)
        transition_kimg = int(self.cfg.train.transition_kimg * 1000)

        for R in range(from_level-1, to_level):
            batch_size = self.bs_map[2 ** (R+1)]

            phases = {'stabilize':[0, train_kimg//batch_size], 'fade_in':[train_kimg//batch_size+1, (transition_kimg+train_kimg)//batch_size]}
            if self.is_restored and R == from_level-1:
                phases[self._phase][0] = self._epoch + 1
                if self._phase == 'fade_in':
                    del phases['stabilize']

            for phase in ['stabilize', 'fade_in']:
                if phase in phases:
                    _range = phases[phase]
                    self.train_phase(R, phase, batch_size, _range[0]*batch_size, _range[0], _range[1])

    def sample(self):
        batch_size = self.z.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        samples = []
        i = j = 0
        for row in range(n_row):
            one_row = []
            # fake
            for col in range(n_col):
                one_row.append(self.fake[i].cpu().data.numpy())
                i += 1
            # real
            for col in range(n_col):
                one_row.append(self.real[j].cpu().data.numpy())
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])

        half = samples.shape[1] // 2
        samples[:, :half, :] = samples[:, :half, :] - np.min(samples[:, :half, :])
        samples[:, :half, :] = samples[:, :half, :] / np.max(samples[:, :half, :])
        samples[:, half:, :] = samples[:, half:, :] - np.min(samples[:, half:, :])
        samples[:, half:, :] = samples[:, half:, :] / np.max(samples[:, half:, :])
        return samples

    def save(self, file_name):
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'
        torch.save(self.G.state_dict(), g_file)
        torch.save(self.D.state_dict(), d_file)

