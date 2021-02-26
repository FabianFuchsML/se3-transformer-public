from utils.utils_profiling import *  # load before other local modules

import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
import torch
import wandb
import time
import datetime

from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.nbody.nbody_dataloader import RIDataset
from utils import utils_logging

from experiments.nbody import nbody_models as models
from equivariant_attention.from_se3cnn.SO3 import rot
from experiments.nbody.nbody_flags import get_flags


def to_np(x):
    return x.cpu().detach().numpy()


def get_acc(pred, x_T, v_T, y=None, verbose=True):

    acc_dict = {}
    pred = to_np(pred)
    x_T = to_np(x_T)
    v_T = to_np(v_T)
    assert len(pred) == len(x_T)

    if verbose:
        y = np.asarray(y.cpu())
        _sq = (pred - y) ** 2
        acc_dict['mse'] = np.mean(_sq)

    _sq = (pred[:, 0, :] - x_T) ** 2
    acc_dict['pos_mse'] = np.mean(_sq)

    _sq = (pred[:, 1, :] - v_T) ** 2
    acc_dict['vel_mse'] = np.mean(_sq)

    return acc_dict


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, schedul, FLAGS):
    model.train()
    loss_epoch = 0

    num_iters = len(dataloader)
    wandb.log({"lr": optimizer.param_groups[0]['lr']}, commit=False)
    for i, (g, y1, y2) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        x_T = y1.to(FLAGS.device).view(-1, 3)
        v_T = y2.to(FLAGS.device).view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(g)
        loss = loss_fnc(pred, y)
        loss_epoch += to_np(loss)

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        # backprop
        loss.backward()
        optimizer.step()

        # print to console
        if i % FLAGS.print_interval == 0:
            print(
                f"[{epoch}|{i}] loss: {loss:.5f}")

        # log to wandb
        if i % FLAGS.log_interval == 0:
            # 'commit' is only set to True here, meaning that this is where
            # wandb counts the steps
            wandb.log({"Train Batch Loss": to_np(loss)}, commit=True)

        # exit early if only do profiling
        if FLAGS.profile and i == 10:
            sys.exit()

        schedul.step(epoch + i / num_iters)

    # log train accuracy for entire epoch to wandb
    loss_epoch /= len(dataloader)
    wandb.log({"Train Epoch Loss": loss_epoch}, commit=False)


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS, dT):
    model.eval()

    keys = ['pos_mse', 'vel_mse']
    acc_epoch = {k: 0.0 for k in keys}
    acc_epoch_blc = {k: 0.0 for k in keys}  # for constant baseline
    acc_epoch_bll = {k: 0.0 for k in keys}  # for linear baseline
    loss_epoch = 0.0
    for i, (g, y1, y2) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        x_T = y1.view(-1, 3)
        v_T = y2.view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1).to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()
        loss_epoch += to_np(loss_fnc(pred, y)/len(dataloader))
        acc = get_acc(pred, x_T, v_T, y=y)
        for k in keys:
            acc_epoch[k] += acc[k]/len(dataloader)

        # eval constant baseline
        bl_pred = torch.zeros_like(pred)
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_blc[k] += acc[k]/len(dataloader)

        # eval linear baseline
        # Apply linear update to locations.
        bl_pred[:, 0, :] = dT * g.ndata['v'][:, 0, :]
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_bll[k] += acc[k] / len(dataloader)

    print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")
    wandb.log({"Test loss": loss_epoch}, commit=False)
    for k in keys:
        wandb.log({"Test " + k: acc_epoch[k]}, commit=False)
    wandb.log({'Const. BL pos_mse': acc_epoch_blc['pos_mse']}, commit=False)
    wandb.log({'Linear BL pos_mse': acc_epoch_bll['pos_mse']}, commit=False)
    wandb.log({'Linear BL vel_mse': acc_epoch_bll['vel_mse']}, commit=False)


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs, y1, y2 = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(y1), torch.stack(y2)


def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    train_dataset = RIDataset(FLAGS, split='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=FLAGS.num_workers,
                              drop_last=True)

    test_dataset = RIDataset(FLAGS, split='test')
    # drop_last is only here so that we can count accuracy correctly;
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False,
                             collate_fn=collate,
                             num_workers=FLAGS.num_workers,
                             drop_last=True)

    # time steps
    assert train_dataset.data['delta_T'] == test_dataset.data['delta_T']
    assert train_dataset.data['sample_freq'] == test_dataset.data['sample_freq']
    print(f'deltaT: {train_dataset.data["delta_T"]}, '
          f'freq: {train_dataset.data["sample_freq"]}, '
          f'FLAGS.ri_delta_t: {FLAGS.ri_delta_t}')
    dT = train_dataset.data['delta_T'] * train_dataset.data[
        'sample_freq'] * FLAGS.ri_delta_t

    FLAGS.train_size = len(train_dataset)
    FLAGS.test_size = len(test_dataset)
    assert len(test_dataset) < len(train_dataset)

    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, FLAGS.num_channels, num_degrees=FLAGS.num_degrees,
                                             div=FLAGS.div, n_heads=FLAGS.head, si_m=FLAGS.simid, si_e=FLAGS.siend,
                                             x_ij=FLAGS.xij)

    utils_logging.write_info_file(model, FLAGS=FLAGS, UNPARSED_ARGV=UNPARSED_ARGV, wandb_log_dir=wandb.run.dir)

    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, FLAGS.num_epochs, eta_min=1e-4)
    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        test_epoch(epoch, model, task_loss, test_loader, FLAGS, dT)


if __name__ == '__main__':

    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Log all args to wandb
    wandb.init(project='equivariant-attention', name=FLAGS.name, config=FLAGS)
    wandb.save('*.txt')

    # Where the magic is
    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback
        traceback.print_exc()
        pdb.post_mortem()
