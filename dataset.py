import argparse
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
from transforms import calc_rot_mat
from utils import get_num_digits


class GraphDataset(Dataset):
    def __init__(self, root, dataset_name, atom_embed,
                 split, cutoff, split_ratio=None, seed=0, use_invariance=False):
        super().__init__()

        if split_ratio is None:
            split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1, }

        assert split in ['train', 'val', 'test'], "Split must be one of ['train', 'val', 'test']!"

        self.root = root
        self.atom_embed = atom_embed
        self.cutoff = cutoff
        self.use_invariance = use_invariance
        self.metadata_dir = osp.join(self.root, 'processed', dataset_name)

        invariance_type = 'invariant' if self.use_invariance else 'default'
        self.processed_dir = osp.join(self.root, 'processed', dataset_name, invariance_type)

        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        if len(glob.glob(osp.join(self.processed_dir, '*.pth'))) !=\
                len(glob.glob(f'{osp.join(self.root, "raw", dataset_name)}/*')):
            print(f'Processed data missing or incomplete. Creating data at {self.processed_dir}...')
            self.process_data(dataset_name)

        self.data_files = sorted(glob.glob(osp.join(self.processed_dir, '*.pth')))

        np.random.seed(seed)
        np.random.shuffle(self.data_files)

        idx = [int(split_ratio['train'] * len(self.data_files)),
               int(split_ratio['val'] * len(self.data_files))]

        if split == 'train':
            self.data_files = self.data_files[: idx[0]]
        elif split == 'val':
            self.data_files = self.data_files[idx[0]: idx[0] + idx[1]]
        else:
            self.data_files = self.data_files[idx[0] + idx[1]:]

        with open(osp.join(self.metadata_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

    def process_data(self, dataset_name):
        with open(self.atom_embed, 'r') as f:
            embed = json.load(f)

        sum_num_edges = 0
        sum_num_nodes = 0
        metadata = {'targets': []}
        for index, file_name in enumerate(tqdm(sorted(glob.glob(f'{osp.join(self.root, "raw", dataset_name)}/*')))):
            with open(file_name, 'r') as f:
                data = json.load(f)

            pos = torch.tensor(data['coordinates'], dtype=torch.float)
            x = torch.tensor([embed[atom] for atom in data['elements']], dtype=torch.float)

            targets = []
            for target in data['targets']:
                target['data'] = torch.tensor(target['data'], dtype=torch.float)
                assert target['level'] in ['graph', 'node'], "Target level must be one of ['graph', 'node']!"
                if target['level'] == 'graph':
                    target['data'] = target['data'].reshape(1, -1)
                else:
                    target['data'] = target['data'].reshape(len(data['elements']), -1)
                targets.append(target)

            if self.use_invariance:
                rot_mat = calc_rot_mat(pos)
                pos = pos @ rot_mat
                for target in targets:
                    if target['isSpatial']:
                        assert target['data'].shape[1] == 3, 'Spatial data must be three-dimensional!'
                        target['data'] = target['data'] @ rot_mat

            graph = Data(x=x, pos=pos)

            for target in targets:
                graph[target['name']] = target['data']

            graph['edge_index'] = radius_graph(graph.pos, self.cutoff)

            src, dst = graph.edge_index
            displacement = graph.pos[dst] - graph.pos[src]
            distance = (displacement ** 2).sum(dim=1, keepdim=True).sqrt()
            graph['ex_norm_displacement'] = torch.cat([displacement / distance, distance / self.cutoff], dim=1)

            if index == 0:
                for target in targets:
                    meta_targets = {key: value for key, value in target.items() if key != 'data'}
                    meta_targets['dim'] = target['data'].shape[1]
                    metadata['targets'].append(meta_targets)

            torch.save(graph, osp.join(self.processed_dir, file_name.split('/')[-1].replace('json', 'pth')))

            sum_num_edges += graph.edge_index.shape[1]
            sum_num_nodes += graph.x.shape[0]

        metadata['averageEdgesPerNode'] = sum_num_edges / sum_num_nodes
        with open(osp.join(self.metadata_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return torch.load(self.data_files[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--cutoff', default=4, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--remake', action='store_true')
    args = parser.parse_args()

    if args.remake:
        for invariance_type in ['default', 'invariant']:
            shutil.rmtree(osp.join('data', 'processed', args.dataset_name, invariance_type), ignore_errors=True)

    datasets = []
    for use_invariance in [True, False]:
        for split in ['train', 'val', 'test']:
            dataset = GraphDataset(root='data',
                                   dataset_name=args.dataset_name,
                                   atom_embed='data/atom_init_embedding.json',
                                   split=split,
                                   cutoff=args.cutoff,
                                   seed=args.seed,
                                   use_invariance=use_invariance)
            datasets.append(dataset)
