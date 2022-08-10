import glob
import json
import os
import os.path as osp
import pdb
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from tqdm import tqdm
from transforms import get_invariant_pos


class GraphDataset(Dataset):
    def __init__(self, root, dataset_name, ele_label, atom_embed,
                 radii, split, cutoff, split_ratio=None, seed=0, use_invariance=False):
        super().__init__()

        if split_ratio is None:
            split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1, }
        self.split_ratio = split_ratio
        self.split = split
        self.seed = seed
        self.root = root
        self.dataset_name = dataset_name
        self.ele_label = ele_label
        self.atom_embed = atom_embed
        self.radii = radii
        self.cutoff = cutoff
        self.use_invariance = use_invariance

        if self.use_invariance:
            self.processed_data_dir = osp.join(self.root, 'processed', 'invariant')
            self.separate_data_dir = osp.join(self.root, 'separate', 'invariant', dataset_name, split)
        else:
            self.processed_data_dir = osp.join(self.root, 'processed', 'default')
            self.separate_data_dir = osp.join(self.root, 'separate', 'default', dataset_name, split)

        self.metadata = {}

        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.separate_data_dir, exist_ok=True)

        if len(os.listdir(self.separate_data_dir)) == 0:
            processed_data_path = osp.join(self.processed_data_dir, f'{self.dataset_name}_data.pth')
            if not osp.exists(processed_data_path):
                data = self.get_data(self.dataset_name)
                sum_num_edges = 0
                sum_num_nodes = 0
                for graph in data:
                    sum_num_edges += graph.edge_index.shape[1]
                    sum_num_nodes += graph.x.shape[0]
                avg_edges = sum_num_edges / sum_num_nodes
                print(f'Average incoming edges per node: {avg_edges}')
            else:
                data = torch.load(processed_data_path)

            np.random.seed(seed)
            np.random.shuffle(data)

            idx = [int(self.split_ratio['train'] * len(data)),
                   int(self.split_ratio['val'] * len(data))]

            if self.split == 'train':
                data = data[: idx[0]]
            elif self.split == 'val':
                data = data[idx[0]: idx[0] + idx[1]]
            else:
                data = data[idx[0] + idx[1]:]

            print(f'Storing separate graphs from {self.split} set...')
            for i, graph in enumerate(tqdm(data)):
                torch.save(graph, osp.join(self.separate_data_dir, f'{i:06d}.pth'))

            with open(osp.join(self.processed_data_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)

            sum_num_edges = 0
            sum_num_nodes = 0
            for graph in data:
                sum_num_edges += graph.edge_index.shape[1]
                sum_num_nodes += graph.x.shape[0]
            self.metadata['averageEdgesPerNode'] = sum_num_edges / sum_num_nodes

            with open(osp.join(self.separate_data_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=4)
        else:
            with open(osp.join(self.separate_data_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)

    def get_data(self, dataset_name):
        with open(self.atom_embed, 'r') as f:
            embed = json.load(f)

        with open(self.radii, 'r') as f:
            radii = json.load(f)

        with open(self.ele_label, 'r') as f:
            ele_label = json.load(f)

        graph_data = []
        for index, file_name in enumerate(tqdm(sorted(glob.glob(f'{osp.join(self.root, "raw", dataset_name)}/*')))):
            with open(file_name, 'r') as f:
                data = json.load(f)

            pos = torch.tensor(data['coordinates'], dtype=torch.float)
            x = torch.tensor([embed[atom] for atom in data['elements']], dtype=torch.float)

            targets = []
            for target in data['targets']:
                target['data'] = torch.tensor(target['data'], dtype=torch.float)
                if target['level'] == 'graph':
                    target['data'] = target['data'].reshape(1, -1)
                elif target['level'] == 'node':
                    target['data'] = target['data'].reshape(len(atom_list), -1)
                else:
                    raise Exception(f"Unrecognized level: {target['level']}. It must be either 'graph' or 'node'.")
                targets.append(target)

            if self.use_invariance:
                trans_result = get_invariant_pos(pos)
                pos = torch.from_numpy(trans_result['trans']).float()
                rot_mat = torch.from_numpy(trans_result['rot_mat']).float()
                for target in targets:
                    if target['isSpatial']:
                        assert target['data'].shape(1) == 3, 'Spatial data must be three-dimensional!'
                        target['data'] = target['data'] @ rot_mat

            graph = Data(x=x, pos=pos)

            for target in targets:
                graph[target['name']] = target['data']

            graph.edge_index = radius_graph(graph.pos, self.cutoff)

            src, dst = graph.edge_index
            displacement = graph.pos[dst] - graph.pos[src]

            distance = (displacement ** 2).sum(dim=1, keepdim=True).sqrt()
            n_st = displacement / distance

            if n_st.isnan().sum() > 0:
                print(f'File {file_name} encountered null! Skipped.')
                continue

            atom_radii = torch.tensor([radii[atom] for atom in data['elements']], dtype=torch.float).reshape(-1, 1)
            a_s = atom_radii[src]
            a_t = atom_radii[dst]
            p_st = torch.cat([distance, distance - a_s, distance - a_t, distance - a_s - a_t], dim=1)

            graph.edge_attr = torch.cat([n_st, p_st / self.cutoff], dim=1)

            if index == 0:
                self.metadata['targets'] = []
                for target in targets:
                    meta_targets = {key: value for key, value in target.items() if key != 'data'}
                    meta_targets['dim'] = target['data'].shape[1]
                    self.metadata['targets'].append(meta_targets)
                    with open(osp.join(self.processed_data_dir, 'metadata.json'), 'w') as f:
                        json.dump(self.metadata, f, indent=4)

            graph_data.append(graph)
        torch.save(graph_data, osp.join(self.processed_data_dir, f'{dataset_name}_data.pth'))
        return graph_data

    def __len__(self):
        return len(glob.glob(f'{self.separate_data_dir}/*.pth'))

    def __getitem__(self, idx):
        return torch.load(osp.join(self.separate_data_dir, f'{idx:06d}.pth'))

    def get_metadata(self):
        return self.metadata


if __name__ == '__main__':
    datasets = []
    for use_invariance in [True, False]:
        for split in ['train', 'val', 'test']:
            dataset = GraphDataset(root='data',
                                   dataset_name='qm7',
                                   ele_label='data/ele.json',
                                   atom_embed='data/atom_init_embedding.json',
                                   radii='data/radii.json',
                                   split=split,
                                   cutoff=6,
                                   use_invariance=use_invariance)
            datasets.append(dataset)
    pdb.set_trace()
