import dgl
import numpy as np
import torch

from torch import FloatTensor
from torch.utils.data import Dataset

from experiments.toy_optimisation.opt_potential import update_potential_values
from utils.utils_data import update_relative_positions

DTYPE = np.float32
DTYPE_TORCH = torch.float32


class OptDataset(Dataset):
    def __init__(self, FLAGS, split):
        """Create a dataset of graphs. Each graph represents a set of points in 3D space, where each pair of points
        interacts according to a randomly parameterised potential. The parameter(s) of this potential are stored in the
        graph as edge information.

        Node data has shape (num_points, num_channels, data_dimensionality)
        Edge data has shape (num_edges, num_channels, data_dimensionality)
        The node indices on the graphs are ascending-ordered by destination node followed by source node. Other code
        may depend on this ordering."""
        self.n_points = FLAGS.n_points
        self.len = FLAGS.epoch_length
        self.split = split

        assert self.split in ["log", "train"]

    @property
    def _num_directed_edges(self):
        return self.n_points * (self.n_points - 1)

    # noinspection PyArgumentList
    @staticmethod
    def _generate_potential_parameters():
        return FloatTensor(1).uniform_(0.0, 1.0)

    def _generate_graph_edges_with_parameters(self):
        """Generate source node indices, destination node indices, and potential parameters for a fully connected graph.

        The returned indices are ascending-ordered by destination node followed by source node."""
        src = []
        dst = []
        list_of_parameters = []

        parameters_dict = {}

        for i in range(self.n_points):
            for j in range(self.n_points):
                key = frozenset({i, j})

                if i != j:
                    if key not in parameters_dict.keys():
                        parameters_dict[key] = self._generate_potential_parameters()

                    # Add indices and parameters for the j -> i edge.
                    dst.append(i)
                    src.append(j)
                    list_of_parameters.append(parameters_dict[key])

        parameters_shape = (self._num_directed_edges, 1, 1)
        parameters_tensor = torch.tensor(list_of_parameters).reshape(parameters_shape)

        return np.array(src), np.array(dst), parameters_tensor

    @staticmethod
    def _get_random_coordinates(n_points):
        loc_std = 0.5
        random_coordinates = torch.randn(size=(n_points, 3), dtype=DTYPE_TORCH) * loc_std
        return torch.unsqueeze(random_coordinates, dim=1)  # Must have shape [N, 1, 3]

    # noinspection PyTypeChecker
    def __getitem__(self, idx):
        if self.split == 'log':
            torch.manual_seed(idx)

        x = self._get_random_coordinates(self.n_points)
        indices_src, indices_dst, potential_parameters = self._generate_graph_edges_with_parameters()

        G = dgl.DGLGraph((indices_src, indices_dst))

        G.ndata['x'] = x
        G.ndata['ones'] = torch.ones(size=[self.n_points, 1, 1], dtype=DTYPE_TORCH)  # used as dummy input for network
        G.edata['potential_parameters'] = potential_parameters

        update_relative_positions(G)
        update_potential_values(G)

        return G

    def __len__(self):
        return self.len
