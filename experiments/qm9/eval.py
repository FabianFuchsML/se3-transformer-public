from utils.utils_profiling import * # load before other local modules

import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import math
import numpy as np
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from QM9 import QM9Dataset

from experiments.qm9 import models #as models

def to_np(x):
    return x.cpu().detach().numpy()

def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.test_size

    print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)

def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    test_dataset = QM9Dataset(FLAGS.data_address, 
                             FLAGS.task, 
                             mode='test') 
    test_loader = DataLoader(test_dataset, 
                             batch_size=FLAGS.batch_size, 
			     shuffle=False, 
                             collate_fn=collate, 
                             num_workers=FLAGS.num_workers)

    FLAGS.test_size = len(test_dataset)

    # Choose model
    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, 
                                             train_dataset.atom_feature_size, 
                                             FLAGS.num_channels,
                                             num_nlayers=FLAGS.num_nlayers,
                                             num_degrees=FLAGS.num_degrees,
                                             edge_dim=train_dataset.num_bonds,
                                             div=FLAGS.div,
                                             pooling=FLAGS.pooling,
                                             n_heads=FLAGS.head)
    if not FLAGS.restore:
        print("FLAGS.restore must be set")
    model.load_state_dict(torch.load(FLAGS.restore))
    print(f"Model loaded from {FLAGS.restore}")
    model.to(FLAGS.device)

    # Loss function
    def task_loss(pred, target, use_mean=True):
        l1_loss = torch.sum(torch.abs(pred - target))
        l2_loss = torch.sum((pred - target)**2)
        if use_mean:
            l1_loss /= pred.shape[0]
            l2_loss /= pred.shape[0]

        rescale_loss = train_dataset.norm2units(l1_loss, FLAGS.task)
        return l1_loss, l2_loss, rescale_loss

    # Save path

    # Run training
    print('Begin evaluation')
    test_epoch(0, model, task_loss, test_loader, FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3Transformer', 
            help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
            help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
            help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=16,
            help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0,
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4,
            help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg',
            help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1,
            help="Number of attention heads")

    # Data
    parser.add_argument('--data_address', type=str, default='QM9_data.pt',
            help="Address to structure file")
    parser.add_argument('--task', type=str, default='homo',
            help="QM9 task ['homo, 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']")

    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--restore', type=str, default=None,
            help="Path to model to restore")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4, 
            help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
            help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{FLAGS.num_degrees}-l{FLAGS.num_layers}-{FLAGS.num_channels}-{FLAGS.num_nlayers}'

    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    main(FLAGS, UNPARSED_ARGV)
