import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3Transformer',
                        help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
                        help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=4,
                        help="Number of channels in middle layers")
    parser.add_argument('--div', type=float, default=1,
                        help="Low dimensional embedding fraction")
    parser.add_argument('--head', type=int, default=1,
                        help="Number of attention heads")

    # Type of self-interaction in attention layers,
    # valid: '1x1' (simple) and 'att' (attentive) with a lot more parameters
    parser.add_argument('--simid', type=str, default='1x1',)
    parser.add_argument('--siend', type=str, default='att')
    parser.add_argument('--xij', type=str, default='add')

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=500,
                        help="Number of epochs")

    # Data
    # An argument to specify which dataset type to use (for now)
    parser.add_argument('--ri_data_type', type=str, default="charged",
                        choices=['charged', 'charged_infer', 'springs',
                                 'springs_infer'])
    # location of data for relational inference
    parser.add_argument('--ri_data', type=str, default='data_generation')
    parser.add_argument('--data_str', type=str, default='my_datasetfile')
    # how many time steps to predict into the future
    parser.add_argument('--ri_delta_t', type=int, default=10)
    # how many time steps to cut off from dataset in the beginning
    parser.add_argument('--ri_burn_in', type=int, default=0)
    parser.add_argument('--ri_start_at', type=str, default='all')

    # Logging
    parser.add_argument('--name', type=str, default='ri_dgl', help="Run name")
    parser.add_argument('--log_interval', type=int, default=25,
                        help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
                        help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
                        help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
                        help="Path to model to restore")
    parser.add_argument('--verbose', type=int, default=0)

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
                        help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=1992)

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
