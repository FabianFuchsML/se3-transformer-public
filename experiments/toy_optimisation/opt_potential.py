import torch

from equivariant_attention.modules import get_r


def potential_function(r, potential_parameters):
    assert torch.is_tensor(r)
    x = r - potential_parameters - 1
    potential_function_global_min = -0.321919
    return x**4 - x**2 + 0.1*x - potential_function_global_min


def potential_gradient(r, potential_parameters):
    assert torch.is_tensor(r)
    x = r - potential_parameters - 1
    return 4*x**3 - 2*x + 0.1


def apply_potential_function(edge):
    potential_parameters = edge.data['potential_parameters']
    # The key `w` is already used as a type-0 edge feature in
    # equivariant_attention/modules.py, so we reuse the key here.
    return {'w': potential_function(edge.data['r'], potential_parameters)}


def update_potential_values(G, r=None):
    """For each directed edge in the graph, compute the value of the potential between the source and destination nodes.
    Write the computed potential values to the graph as edge data."""
    if r is None:
        r = get_r(G)
    G.edata['r'] = r
    G.apply_edges(func=apply_potential_function)


def compute_overall_potential(G):
    return torch.mean(G.edata['w'])
