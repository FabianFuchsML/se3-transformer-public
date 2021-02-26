from utils.utils_profiling import *  # Load before other local modules

import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
import dgl
import numpy as np
import torch
import wandb

from torch import optim
from torch.utils.data import DataLoader
from experiments.toy_optimisation.opt_dataloader import OptDataset
from experiments.toy_optimisation.opt_models import GradientDescent
from experiments.toy_optimisation.opt_potential import compute_overall_potential

from utils import utils_logging
from utils.utils_data import PickleGraph, to_np

from experiments.toy_optimisation import opt_models as models
from experiments.toy_optimisation.opt_flags import get_flags


def train_epoch(epoch, model, loss_function, dataloader, optimizer, schedule, FLAGS):
    model.train()

    loss_log = []
    tuned_loss_log = []
    update_norm_log = []
    tuned_update_norm_log = []
    update_norm_steps_log = {i: [] for i in range(FLAGS.num_iter)}

    gradient_descent_model = GradientDescent(step_size=FLAGS.gd_stepsize,
                                             convergence_tolerance=FLAGS.gd_tolerance)

    num_iters = len(dataloader)
    wandb.log({"lr": optimizer.param_groups[0]['lr']}, commit=False)

    for i, g in enumerate(dataloader):
        g = g.to(FLAGS.device)

        optimizer.zero_grad()
        pred, _ = model(g)
        loss = loss_function(pred)
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if total_norm == total_norm:
            optimizer.step()
        else:
            warnings.warn('Gradient update skipped because of NaNs')

        loss_log.append(to_np(loss))
        update_norm_log.append(np.mean(to_np(pred.ndata['update_norm'])))
        if 'update_norm_0' in pred.ndata and FLAGS.num_iter > 1:
            for j in range(FLAGS.num_iter):
                update_norm_steps_log[j].append(np.mean(to_np(pred.ndata[f'update_norm_{j}'])))

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

        if i % FLAGS.log_interval == 0:
            wandb.log({"Train Batch Loss": to_np(loss)}, commit=True)

        # Gradient descent post-processing
        if FLAGS.gd_post_process:
            with torch.no_grad():
                tuned_pred, _ = gradient_descent_model(pred)
                tuned_loss = loss_function(tuned_pred)
                tuned_loss_log.append(to_np(tuned_loss))

                tuned_update_norm_log.append(np.mean(to_np(tuned_pred.ndata['update_norm'])))

                if i % FLAGS.print_interval == 0:
                    print(f"[{epoch}|{i}] tuned loss: {to_np(tuned_loss):.5f}")

                if i % FLAGS.log_interval == 0:
                    wandb.log({"Train Batch Tuned Loss": to_np(tuned_loss)}, commit=False)

        # Exit early if only do profiling
        if FLAGS.profile and i == 10:
            sys.exit()

        schedule.step(epoch + i / num_iters)

    # Gradient logging
    basis_gradients = utils_logging.get_average('Basis computation flow')
    utils_logging.clear_data('Basis computation flow')
    wandb.log({"Average gradient norm: Basis": basis_gradients}, commit=False)

    network_gradients = utils_logging.get_average('Neural networks flow')
    utils_logging.clear_data('Neural networks flow')
    wandb.log({"Average gradient norm: Neural nets": network_gradients}, commit=False)

    # Loss logging
    average_loss = np.mean(np.array(loss_log))
    wandb.log({"Train Epoch Loss": average_loss}, commit=False)

    """
    A note about 'update norms':
    
    We log both the overall average size of the update steps in coordinate space as well as the intermediate steps.
    The length of the intermediate steps for the iterative version may seem surpsingly large given the smaller overall
    update. We analysed this in debugging mode and found no inconsistencies. Our current interpretation is that the
    network 'samples' different configurations to explore the effect and interplay of different pairwise potentials.    
    """

    average_update_norm = np.mean(np.array(update_norm_log))
    wandb.log({"Average Update Norm": average_update_norm}, commit=False)

    for j in range(FLAGS.num_iter):
        avg_steps = np.mean(np.array(update_norm_steps_log[j]))
        wandb.log({f"Average Update Norm {j}": avg_steps}, commit=False)

    if FLAGS.gd_post_process:
        average_tuned_loss = np.mean(np.array(tuned_loss_log))
        wandb.log({"Train Epoch Tuned Loss": average_tuned_loss}, commit=False)

        average_tuned_update_norm = np.mean(np.array(tuned_update_norm_log))
        wandb.log({"Average Tuned Update Norm": average_tuned_update_norm}, commit=False)


def run_forward_pass(model, *, dataloader, loss_function):
    if hasattr(model, 'eval'):
        model.eval()

    losses = []
    update_norms = []
    update_norm_steps = {i: [0] for i in range(FLAGS.num_iter)}

    for i, g in enumerate(dataloader):
        g = g.to(FLAGS.device)

        pred, _ = model(g)
        loss = loss_function(pred)
        losses.append(to_np(loss))

        update_norms.append(np.mean(to_np(pred.ndata['update_norm'])))

        if not isinstance(model, GradientDescent):
            for j in range(FLAGS.num_iter):
                update_norm_steps[j].append(np.mean(to_np(pred.ndata[f'update_norm_{j}'])))

        if i % FLAGS.print_interval == 0:
            print(f"[0|{i}] loss: {loss:.5f}")

        wandb.log({"Train Batch Loss": to_np(loss)}, commit=True)

        # Exit early if only do profiling
        if FLAGS.profile and i == 10:
            sys.exit()

    average_loss = np.mean(np.array(losses))
    wandb.log({"Train Epoch Loss": average_loss}, commit=False)

    average_update_norm = np.mean(np.array(update_norms))
    wandb.log({"Average Update Norm": average_update_norm}, commit=False)

    for j in range(FLAGS.num_iter):
        avg_steps = np.mean(np.array(update_norm_steps[j]))
        wandb.log({f"Average Update Norm {j}": avg_steps}, commit=False)


def collate(list_of_graphs):
    batched_graph = dgl.batch(list_of_graphs)
    return batched_graph


def log_batch(model, dataloader, FLAGS):
    if hasattr(model, 'eval'):
        model.eval()

    g = next(iter(dataloader))  # Get a single batch
    g = g.to(FLAGS.device)
    pred, g_steps = model(g, save_steps=True)  # One forward pass

    #reate dictionary for storing node and edge information
    edge_keys = ['potential_parameters', 'w', 'r']
    node_keys = ['x']

    log_values = [
        [
            PickleGraph(G, desired_keys=edge_keys + node_keys)
            for G in dgl.unbatch(batched_graph)
        ]
        for batched_graph in g_steps
    ]

    filename = os.path.join(wandb.run.dir, "optimisation_steps.pkl")
    with open(filename, "wb") as file:
        pickle.dump(log_values, file)


def drop_to_debugger_or_dump_exception(FLAGS):
    import traceback
    if FLAGS.num_runs == 1:
        import pdb
        traceback.print_exc()
        pdb.post_mortem()
    else:
        with open(os.path.join(wandb.run.dir, "exception.txt"), "w") as file:
            traceback.print_exc(file=file)


def main(FLAGS, UNPARSED_ARGV):
    train_dataset = OptDataset(FLAGS, split='train')
    dataloader = DataLoader(train_dataset,
                            batch_size=FLAGS.batch_size,
                            shuffle=True,
                            collate_fn=collate,
                            num_workers=FLAGS.num_workers,
                            drop_last=True)

    model = models.__dict__.get(FLAGS.model)(
        num_layers=FLAGS.num_layers, num_channels=FLAGS.num_channels, num_degrees=FLAGS.num_degrees,
        div=FLAGS.div, n_heads=FLAGS.head, si_m=FLAGS.simid, si_e=FLAGS.siend, x_ij=FLAGS.xij,
        num_iter=FLAGS.num_iter, compute_gradients=FLAGS.basis_gradients, k_neighbors=FLAGS.k_neighbors,
        step_size=FLAGS.gd_stepsize, convergence_tolerance=FLAGS.gd_tolerance)

    if isinstance(model, torch.nn.Module):
        model.to(FLAGS.device)

    utils_logging.write_info_file(model, FLAGS=FLAGS, UNPARSED_ARGV=UNPARSED_ARGV, wandb_log_dir=wandb.run.dir)

    task_loss = compute_overall_potential

    if FLAGS.restore:
        model.load_state_dict(torch.load(FLAGS.restore))

    if FLAGS.forward or isinstance(model, GradientDescent):
        for epoch in range(FLAGS.num_epochs):
            run_forward_pass(model, dataloader=dataloader, loss_function=task_loss)
    else:
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, FLAGS.num_epochs, eta_min=0.1 * FLAGS.lr)
        save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

        print('Begin training')
        for epoch in range(FLAGS.num_epochs):
            torch.save(model.state_dict(), save_path)
            print(f'Saved: {save_path}')

            train_epoch(epoch, model, task_loss, dataloader, optimizer, scheduler, FLAGS)

    if FLAGS.log_steps:
        log_dataset = OptDataset(FLAGS, split='log')
        log_loader = DataLoader(log_dataset, batch_size=FLAGS.batch_size, shuffle=False, collate_fn=collate)
        log_batch(model, log_loader, FLAGS)


FLAGS, UNPARSED_ARGV = get_flags()

os.makedirs(FLAGS.save_dir, exist_ok=True)

for run in range(FLAGS.num_runs):
    try:
        # Log all args to wandb
        with wandb.init(project='iterativeSE3', name=f'{FLAGS.name}{run:02d}', config=FLAGS, reinit=True):
            wandb.save('*.txt')
            wandb.save('*.pkl')

            try:
                main(FLAGS, UNPARSED_ARGV)
            except Exception:
                drop_to_debugger_or_dump_exception(FLAGS)
    except Exception:
        drop_to_debugger_or_dump_exception(FLAGS)

    FLAGS.seed += 1
