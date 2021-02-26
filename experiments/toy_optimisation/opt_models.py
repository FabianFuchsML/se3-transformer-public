import warnings
from collections.abc import Callable

import dgl
import numpy as np
import torch
from dgl import function as fn
from torch import nn

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, get_r
from experiments.toy_optimisation.opt_potential import potential_gradient, update_potential_values
from utils.utils_data import copy_dgl_graph, update_relative_positions


class SE3TransformerIterative(nn.Module):
    def __init__(self, *, num_layers: int, num_degrees: int = 4, num_channels: int,
                 div: float = 4, n_heads: int = 1, num_iter: int = 3,
                 si_m='1x1', si_e='1x1', x_ij=None,
                 compute_gradients=True, k_neighbors=None, **kwargs):
        """Iterative SE(3) equivariant GCN with attention

        Args:
            num_layers: number of layers per SE3-Transformer block
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            num_channels: number of channels per degree
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            num_iter: number of SE3-Transformer blocks with individual coordinate outputs
            si_m: ['1x1', 'att'] type of self-interaction in hidden layers
            si_e: ['1x1', 'att'] type of self-interaction in final layer
            x_ij: ['add', 'cat'] use relative position as edge feature
            compute_gradients: [True, False] backpropagate through spherical harmonics computation
            k_neighbors: attend to K neighbours with strongest interaction
            kwargs: catch arguments that are not used in this method
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.div = div
        self.n_heads = n_heads
        self.si_m = si_m
        self.si_e = si_e
        self.x_ij = x_ij
        self.num_iter = num_iter
        self.compute_gradients = compute_gradients
        self.k_neighbors = k_neighbors

        self.edge_dim = 1
        self.fibers = {'in': Fiber(dictionary={0: 1}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={1: 1})}

        self._graph_attention_common_params = {
            'edge_dim': self.edge_dim,
            'learnable_skip': True,
            'skip': 'cat',
            'x_ij': self.x_ij
        }

        self.blocks = self._build_gcn(self.fibers)

    def _input_layer(self):
        return GSE3Res(self.fibers['in'], self.fibers['mid'],
                       div=self.div, n_heads=self.n_heads, selfint=self.si_m,
                       **self._graph_attention_common_params)

    def _middle_layer(self):
        return GSE3Res(self.fibers['mid'], self.fibers['mid'],
                       div=self.div, n_heads=self.n_heads, selfint=self.si_m,
                       **self._graph_attention_common_params)

    def _output_layer(self):
        return GSE3Res(self.fibers['mid'], self.fibers['out'],
                       div=1, n_heads=min(self.n_heads, 2), selfint=self.si_e,
                       **self._graph_attention_common_params)

    def _build_gcn(self, fibers):
        outer_block = []

        for i in range(self.num_iter):
            inner_block = []

            for j in range(self.num_layers):
                if i == 0 and j == 0:
                    inner_block.append(self._input_layer())
                else:
                    inner_block.append(self._middle_layer())
                # Non-linearity between layers
                inner_block.append(GNormBias(fibers['mid']))

            if i == self.num_iter - 1:
                inner_block.append(self._output_layer())

            outer_block.append(nn.ModuleList(inner_block))

        return nn.ModuleList(outer_block)

    def forward(self, G, save_steps=False):
        G_steps = []

        # We need to keep a copy of the initial position of all nodes in order to calculate the overall change in
        # position between the input and the output. This information is used for logging purposes.
        original_x = torch.clone(G.ndata['x'])

        # 'features' are the node inputs to every layer; this dataset comes with edge features but without node features
        # hence, in the first layer, we need dummy node features; we choose this to be a single [1] for each node
        features = {'0': G.ndata['ones']}

        for j, inner_block in enumerate(self.blocks):
            if save_steps:
                G_steps.append(copy_dgl_graph(G))  # for logging only

            num_nodes = _get_number_of_nodes_per_batch_element(G)
            if self.k_neighbors is not None and self.k_neighbors < num_nodes:
                network_input_graph = copy_without_weak_connections(G, K=self.k_neighbors)
            else:
                network_input_graph = G

            # Compute equivariant weight basis for the current graph
            basis, r = get_basis_and_r(network_input_graph, self.num_degrees - 1,
                                       compute_gradients=self.compute_gradients)

            if torch.min(r) < 1e-5:
                warnings.warn("Minimum separation between nodes fell below 1e-5")

            for layer in inner_block:
                features = layer(features, G=network_input_graph, r=r, basis=basis)

            # We arbitrarily use the first type-1 feature for the position updates.
            position_updates = features['1'][:, 0:1, :]
            G.ndata['x'] = G.ndata['x'] + position_updates

            update_relative_positions(G)
            update_potential_values(G)

            # Track the updates at each iteration and store on the graph for logging.
            G.ndata[f'update_norm_{j}'] = torch.sqrt(torch.sum(position_updates**2, -1, keepdim=True))

        # Calculate the overall update and store on the graph for logging.
        overall_update = G.ndata['x'] - original_x
        G.ndata['update_norm'] = torch.sqrt(torch.sum(overall_update**2, -1, keepdim=True))

        if save_steps:
            G_steps.append(copy_dgl_graph(G))

        return G, G_steps


class GradientDescent:
    STEPS_LIMIT = 5000

    def __init__(self, step_size=1e-1, convergence_tolerance=1e-5, **kwargs):
        """Gradient descent model.
        TODO: Explain that we clamp the size of the gradients to avoid explosion, and that we only attempt up to 5000 steps

        Args
            convergence_tolerance: float. Gradient descent will terminate if the largest update across all nodes at a
                given step of gradient descent falls below `convergence_tolerance * step_size`. Equivalently, gradient
                descent terminates if the maximum absolute value of the gradient across all nodes is less than
                `convergence_tolerance`.
            step_size: float
        """
        self._step_size = step_size
        self._convergence_tolerance = convergence_tolerance

    def forward(self, G, save_steps=False, *, inside_loop_callback=None):
        G_steps = []
        num_steps = 0

        params = G.edata['potential_parameters']
        initial_x = torch.clone(G.ndata['x'])
        initial_x.requires_grad_()

        converged = False

        for i in range(self.STEPS_LIMIT):
            num_steps += 1
            update_relative_positions(G)
            G.edata['r'] = get_r(G)

            if inside_loop_callback is not None:
                inside_loop_callback(G)

            if save_steps:
                update_potential_values(G)  # only needed if save_steps==True
                G_steps.append(copy_dgl_graph(G))

            update_direction = G.edata['d'] / torch.norm(G.edata['d'], dim=2, keepdim=True)
            update_size = -potential_gradient(G.edata['r'], params) * self._step_size
            update_size = torch.clamp(update_size, -0.1, 0.1)

            G.edata['update'] = update_size * update_direction
            G.update_all(fn.copy_e('update', 'message'), fn.sum('message', 'update'))
            G.ndata['x'] = G.ndata['x'] + G.ndata['update']

            update_norms = torch.norm(G.ndata['update'], dim=2)
            max_update_norm = torch.max(update_norms)

            if max_update_norm < (self._convergence_tolerance * self._step_size):
                if inside_loop_callback is not None:
                    inside_loop_callback(G)
                converged = True
                break

        if not converged:
            print("Gradient Descent failed to converge after 5000 steps.")

        overall_update = G.ndata['x'] - initial_x
        G.ndata['update_norm'] = torch.sqrt(torch.sum(overall_update**2, -1, keepdim=True))

        update_potential_values(G)  # compute regardless of save_steps

        if save_steps:
            G_steps.append(copy_dgl_graph(G))
            num_steps = 10
            indices_to_keep = np.linspace(0, len(G_steps) - 1, num_steps)
            G_steps = [G_steps[int(i)] for i in indices_to_keep]

        return G, G_steps

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def copy_without_weak_connections(G, K):
    """Make a copy of a graph, preserving only the K strongest incoming edges for each node.

    This code depends on the order in which nodes were added when constructing G. Nodes must be ordered by
    destination index, then by source index.

    Args
        G: dgl.DGLGraph, the graph to copy. If G is a batched graph, every element of the batch must have the same
            number of nodes
        K: int, the number of connections to preserve in the output graph
    """
    B = G.batch_size
    N = _get_number_of_nodes_per_batch_element(G)
    w = G.edata['w']
    src, dst = G.all_edges()
    src = src.to(device=w.device)
    dst = dst.to(device=w.device)

    # src, dst and w are all sorted by Batch, dst_node_id, src_node_id
    w = w.view(B * N, (N-1))  # [batch * dst, src]
    dst = dst.view(B * N, (N-1))
    src = src.view(B * N, (N-1))

    # sorting
    w, indices = torch.sort(w, descending=True)
    dst = torch.gather(dst, dim=-1, index=indices)
    src = torch.gather(src, dim=-1, index=indices)

    # take top K
    w = w[:, :K]
    dst = dst[:, :K]
    src = src[:, :K]

    # reshape into 1D
    w = torch.reshape(w, (B * N * K, 1, 1))
    dst = torch.reshape(dst, (B * N * K,)).cpu().detach()
    src = torch.reshape(src, (B * N * K,)).cpu().detach()

    # create new graph with less edges and fill with data
    G2 = dgl.DGLGraph((src, dst))
    G2.edata['w'] = w
    G2.ndata['x'] = G.ndata['x']
    update_relative_positions(G2)

    return G2


def _get_number_of_nodes_per_batch_element(batched_graph):
    """For a batched graph, where each element of the batch is known to have the same number of nodes, return
    the number of nodes per batch element."""
    if isinstance(batched_graph.batch_num_nodes, Callable):
        return batched_graph.batch_num_nodes()[0]
    else:
        return batched_graph.batch_num_nodes[0]
