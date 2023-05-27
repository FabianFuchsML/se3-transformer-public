import os
import sys

import dgl
import numpy as np
import torch
import pickle
import graphein.protein as gp
import torch_geometric
import csv
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset, DataLoader
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.ml import ProteinGraphListDataset, GraphFormatConvertor

class AlphaDataset(Dataset):

    atom_feature_size = 20
    num_bonds = 1

    def __init__(self, 
            immuno_path= '/edward-slow-vol/CPSC_552/immunoai/data/immuno_data_train_IEDB_A0201_HLAseq_2_csv.csv', 
            structures_path= '/edward-slow-vol/CPSC_552/alpha_structure', 
            mode: str='train', 
            transform=None): 
        self.immuno_path = immuno_path
        self.structures_path = structures_path
        self.mode = mode
        self.transform = transform
        self.config = ProteinGraphConfig(**{"node_metadata_functions": [amino_acid_one_hot]})

        self.load_data()
        self.len = len(self.targets)
        print(f"Loaded {mode}-set, source: {self.immuno_path}, length: {len(self)}")
    
    def __len__(self):
        return self.len
    
    def load_data(self):

        self.inputs = []
        self.targets = []
        self.immuno_list = []
        self.sequence_list = []
        with open(self.structures_path + '/mapping.pickle', 'rb') as p:
            mapping = pickle.load(p)
            with open(self.immuno_path, "r") as f:
                reader = csv.reader(f)
                for count, line in enumerate(reader):
                    if count==0:
                        continue
                    count = count - 1

                    peptide = line[0].replace("J", "")
                    sequence = line[1]
                    sequence = sequence + peptide
                    self.sequence_list.append(line[0])

                    enrichment = float(line[2])
                    immuno = float(line[3])

                    x = self.structures_path + "/rank_1_" + mapping[sequence] + ".pdb"
                    if not os.path.isfile(x) or os.stat(x).st_size == 0:
                        continue

                    self.inputs.append(x)
                    self.targets.append(enrichment)
                    self.immuno_list.append(immuno)
    
        self.mean = np.mean(self.targets)
        self.std = np.std(self.targets)


    def get_target(self, idx, normalize=True):
        target = self.targets[idx]
        if normalize:
            target = (target - self.mean) / self.std
        if self.mode =='train':
            return target
        else:
            return target, self.immuno_list[idx]

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to original representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        return x

    def get(self, idx):
        return self.inputs[idx]

    def __getitem__(self, idx):
        # Load node features

        x = self.get(idx)

        g = construct_graph(config=self.config, path= x)

        convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
        # g = gp.extract_subgraph_from_chains(g, ["A","B"])
        g = gp.extract_subgraph_from_chains(g, ["B"])

        data = convertor(g)

        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        start = data['edge_index'][0]
        end = data['edge_index'][1]

        one_hot = [d['amino_acid_one_hot'] for n, d in g.nodes(data=True)]
        one_hot = one_hot[-data.num_nodes:]
        node_features = torch.tensor(one_hot)
        data.x = node_features

        row, col = data.edge_index

        g = dgl.graph((row, col))

        # Augmentation on the coordinates
        if self.transform:
            data['coords'] = self.transform(data['coords']).float()

        g.ndata['x'] = data['coords']
        g.ndata['f'] = torch.unsqueeze(data['x'],2).float()
        g.edata['d'] = (data['coords'][start] - data['coords'][end]).float()
        g.edata['w'] = torch.ones(g.edata['d'].shape[0],1).float() # all edges are of the same type

        # Load target
        y = self.get_target(idx, normalize=True)
        y = np.array([y])

        if self.mode =='train':
            return g, y
        else:
            return g, y, self.sequence_list[idx]
        


if __name__ == "__main__":
    def collate(samples):
        graphs, y = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(y)

    # dataset = AlphaDataset()
    dataset = AlphaDataset(mode='test',
                            immuno_path='/edward-slow-vol/CPSC_552/immunoai/data/immuno_data_test_IEDB_A0201_HLAseq_2_csv.csv',
                            structures_path='/edward-slow-vol/CPSC_552/alpha_structure_test') 
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate)

    for data in dataloader:
        print("MINIBATCH")
        break
        # print(data)
