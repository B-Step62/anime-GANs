# -*- coding: utf-8 -*-
import sys, os, time
import argparse
import numpy as np

import torch

sys.path.append(os.pardir)
from pggan import PGGAN
from common.utils.config import Config
from dataset.dataset import FaceDataset
from utils.randomnoisegenerator import RandomNoiseGenerator
from models.model import Generator, Discriminator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', default=0, type=int, help='gpu to use.')
    parser.add_argument('--resume', default=None)
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

    G = Generator(model_cfg=cfg.models.generator, target_size=cfg.train.target_size, xpu=args.gpu)
    D = Discriminator(model_cfg=cfg.models.discriminator, target_size=cfg.train.target_size, xpu=args.gpu)
    #print(G)
    #print(D)
    dataset = FaceDataset(cfg.train.dataset)
    assert len(dataset) > 0
    print(f'train dataset contains {len(dataset)} images.')
    z_generator = RandomNoiseGenerator(cfg.models.generator.z_dim, 'gaussian')
    pggan = PGGAN(G, D, dataset, z_generator, args.gpu, cfg, args.resume)
    pggan.train()

if __name__ == '__main__':
    train()
