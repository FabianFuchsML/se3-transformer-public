import sys
import dgl
import numpy as np
import torch
import os
import pickle
import time

DTYPE = np.float32


class RIDataset(torch.utils.data.Dataset):

    node_feature_size = 1

    def __init__(self, FLAGS, split):
        """Create a dataset object"""

        # Data shapes:
        #   edges :: [samples, bodies, bodies]
        #  points :: [samples, frame, bodies, 3]
        #     vel :: [samples, frame, bodies, 3]
        # charges :: [samples, bodies]
        #   clamp :: [samples, frame, bodies]

        self.FLAGS = FLAGS
        self.split = split

        # Dependent on simulation type set filenames.
        if 'charged' in FLAGS.ri_data_type:
            _data_type = 'charged'
        else:
            assert 'springs' in FLAGS.ri_data_type
            _data_type = 'springs'

        assert split in ["test", "train"]
        filename = 'ds_' + split + '_' + _data_type + '_3D_' + FLAGS.data_str
        filename = os.path.join(FLAGS.ri_data, filename + '.pkl')

        time_start = time.time()
        data = {}
        with open(filename, "rb") as file:
            data = pickle.load(file)

        data["points"] = np.swapaxes(data["points"], 2, 3)[:, FLAGS.ri_burn_in:]
        data["vel"] = np.swapaxes(data["vel"], 2, 3)[:, FLAGS.ri_burn_in:]

        if 'sample_freq' not in data.keys():
            data['sample_freq'] = 100
            data['delta_T'] = 0.001
            print('warning: sample_freq not found in dataset')

        self.data = data
        self.len = data['points'].shape[0]
        self.n_frames = data['points'].shape[1]
        self.n_points = data['points'].shape[2]

        if split == 'train':
            print(data["points"][0, 0, 0])
            print(data["points"][-1, 30, 0])

    # number of instances in the dataset (always need that in a dataset object)
    def __len__(self):
        return self.len

    def connect_fully(self, num_atoms):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        src = []
        dst = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array(src), np.array(dst)

    def __getitem__(self, idx):

        # select a start and a target frame
        if self.FLAGS.ri_start_at == 'zero':
            frame_0 = 0
        else:
            last_pssbl = self.n_frames - self.FLAGS.ri_delta_t
            if 'vic' in self.FLAGS.data_str:
                frame_0 = 30
            elif self.split == 'train':
                frame_0 = np.random.choice(range(last_pssbl))
            elif self.FLAGS.ri_start_at == 'center':
                frame_0 = int((last_pssbl) / 2)
            elif self.FLAGS.ri_start_at == 'all':
                frame_0 = int(last_pssbl/self.len*idx)
        frame_T = frame_0 + self.FLAGS.ri_delta_t  # target frame

        x_0 = torch.tensor(self.data['points'][idx, frame_0].astype(DTYPE))
        x_T = torch.tensor(self.data['points'][idx, frame_T].astype(DTYPE)) - x_0
        v_0 = torch.tensor(self.data['vel'][idx, frame_0].astype(DTYPE))
        v_T = torch.tensor(self.data['vel'][idx, frame_T].astype(DTYPE)) - v_0
        charges = torch.tensor(self.data["charges"][idx].astype(DTYPE))

        # Create graph (connections only, no bond or feature information yet)
        indices_src, indices_dst = self.connect_fully(self.n_points)
        G = dgl.DGLGraph((indices_src, indices_dst))

        ### add bond & feature information to graph
        G.ndata['x'] = torch.unsqueeze(x_0, dim=1)  # [N, 1, 3]
        G.ndata['v'] = torch.unsqueeze(v_0, dim=1)  # [N, 1, 3]
        G.ndata['c'] = torch.unsqueeze(charges, dim=1)  # [N, 1, 1]
        G.edata['d'] = x_0[indices_dst] - x_0[indices_src]  # relative postions
        G.edata['w'] = charges[indices_dst] * charges[indices_src]

        r = torch.sqrt(torch.sum(G.edata['d'] ** 2, -1, keepdim=True))

        return G, x_T, v_T

