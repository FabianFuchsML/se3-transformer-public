import torch
from torch import nn
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, GConvSE3
from equivariant_attention.fibers import Fiber


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers: int, num_channels: int, num_degrees: int = 4, div: float = 4,
                 n_heads: int = 1, si_m='1x1', si_e='att', x_ij='add'):
        """
        Args:
            num_layers: number of attention layers
            num_channels: number of channels per degree
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            si_m: ['1x1', 'att'] type of self-interaction in hidden layers
            si_e: ['1x1', 'att'] type of self-interaction in final layer
            x_ij: ['add', 'cat'] use relative position as edge feature
        """
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 1
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij

        self.fibers = {'in': Fiber(dictionary={1: 1}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={1: 2})}

        self.Gblock = self._build_gcn(self.fibers)
        print(self.Gblock)

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads,
                                  learnable_skip=True, skip='cat', selfint=self.si_m, x_ij=self.x_ij))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim, div=1, n_heads=min(self.n_heads, 2),
                    learnable_skip=True, skip='cat', selfint=self.si_e, x_ij=self.x_ij))
        return nn.ModuleList(Gblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)
        h_enc = {'1': G.ndata['v']}
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc['1']


class TFN(nn.Module):
    """Tensorfiel Network"""

    def __init__(self, num_layers: int, num_channels: int, num_degrees: int = 4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 1

        self.fibers = {'in': Fiber(dictionary={0: 1, 1: 1}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={1: 2})}

        blocks = self._build_gcn(self.fibers)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)
        # purely for counting paramters in utils_logging.py
        self.enc, self.dec = self.Gblock, self.FCblock

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']

        for i in range(self.num_layers-1):
            Gblock.append(GConvSE3(fin, fibers['mid'], self_interaction=True, flavor='TFN', edge_dim=self.edge_dim))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, flavor='TFN', edge_dim=self.edge_dim))

        return nn.ModuleList(Gblock), nn.ModuleList([])

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h_enc = {'0': G.ndata['c'], '1': G.ndata['v']}
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc['1']
