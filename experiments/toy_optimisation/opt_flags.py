import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    _SELF_INTERACTION_HELP = 'Type of self-interaction in {}. Values: \'1x1\' (simple) and \'att\' (attentive). ' \
                             '\'att\' uses more parameters.'

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3TransformerIterative',
                        help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=3,
                        help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=4,
                        help="Number of channels in middle layers")
    parser.add_argument('--num_iter', type=int, default=1,
                        help="Number of SE3 Transformers stacked")
    parser.add_argument('--div', type=float, default=1,
                        help="Low dimensional embedding fraction")
    parser.add_argument('--head', type=int, default=1,
                        help="Number of attention heads")
    parser.add_argument('--simid', type=str, default='1x1',
                        help=_SELF_INTERACTION_HELP.format('middle layers'))
    parser.add_argument('--siend', type=str, default='1x1',
                        help=_SELF_INTERACTION_HELP.format('output layer'))
    parser.add_argument('--xij', type=str, default='add')
    parser.add_argument('--basis_gradients', type=int, default=1)

    # Parameters for the gradient descent baseline
    parser.add_argument('--gd_tolerance', type=float, default=0.01,
                        help="Tolerance for convergence of gradient descent baseline")
    parser.add_argument('--gd_stepsize', type=float, default=0.02,
                        help="Initial step size for gradient descent baseline")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of epochs")
    parser.add_argument('--n_points', type=int, default=10)
    parser.add_argument('--k_neighbors', type=int, default=10)

    # Logging
    parser.add_argument('--epoch_length', type=int, default=5000)
    parser.add_argument('--name', type=str, default='opt', help="Run name")
    parser.add_argument('--log_interval', type=int, default=25,
                        help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
                        help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
                        help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
                        help="Path to model to restore")
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--incl_potential_params', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=0)

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=1992)

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
                        help="Exit after 10 steps for profiling")
    parser.add_argument('--forward', type=int, default=0,
                        help="Run forward pass")
    parser.add_argument('--gd_post_process', type=int, default=1,
                        help="Run gradient descent after the model to fine tune the results")
    parser.add_argument('--num_runs', type=int, default=1,
                        help="Number of experiments to run sequentially")

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    if torch.cuda.is_available():
        FLAGS.device = torch.device('cuda:0')
    else:
        FLAGS.device = torch.device('cpu')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    return FLAGS, UNPARSED_ARGV
