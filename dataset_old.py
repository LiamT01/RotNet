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


# def get_invariant_pos(pos, return_rotation_matrix=True):
#     # Get the center of the point cloud
#     center = pos.mean(0)
#
#     # Shift pos to be centered
#     pos = pos - center
#
#     # Get the first principal component vector
#     _, _, vv = np.linalg.svd(pos)
#     pc1 = vv[0] / np.linalg.norm(vv[0])
#
#     # Project all points onto to pc1
#     # The reference point is the point closest to the center when projected onto pc1
#     projection = np.abs(pos @ pc1.T)
#     ref = np.argmin(projection)
#
#     center2ref = pos[ref]
#
#     # Possible issues here???
#     # If center2ref @ pc1 == 0???
#     # Might consider switch to choosing the point farthest to the center as the reference point???
#
#     # 取最远/最近还有潜在的问题，万一最远/最近点完全对称（其他点不必对称）？
#     # 应该想一个分布函数，计算两边的距离分布特征，然后选。如果两边特征不一样，那么根据一个标准选一边；
#     # 如果两边特征完全一样，说明本身就是对称的，选哪边都一样（但是要考虑原子类型？？比如分布对称但是原子类型不一样？？就不能说是对称的）
#
#     # 或者说干脆用第一主成分和第二主成分，得到一个十字架。然后叉乘一下第三个坐标轴。但需要定一下新坐标系的方向。
#     # 这个时候只要找一个点，新坐标系的取向使得这个点的三个坐标值都是正的。
#     # 可以这样找这个点：
#     # 在原坐标系算距离质心最远的点，然后调整方向使得这个点落在第一象限。
#     # 如果距离为0，只能说明所有点都挤在原点，这种点云太奇葩了，现实中应该不会遇到。
#     # 如果有ties，看能不能break ties。
#
#     # Invert pc1 if the angle > 90 deg
#     if center2ref @ pc1 < 0:
#         pc1 = -pc1
#
#     # Construct axes in a new coordinate system
#     new_x = center2ref / np.linalg.norm(center2ref)
#     new_z = np.cross(new_x, pc1)
#
#     # In rare cases where new_x is parallel to pc1,
#     # choose a vector that is not parallel to new_x,
#     # and create new_z as the cross product of new_x and the chosen vector.
#     if np.linalg.norm(new_z) == 0:
#         bases = np.stack([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         angles = np.arccos(new_x @ bases)
#         chosen_basis = ((angles != 0) & (angles != np.pi)).nonzero()[0][0]
#         new_z = np.cross(new_x, bases[chosen_basis])
#
#     new_z = new_z / np.linalg.norm(new_z)
#     new_y = np.cross(new_z, new_x)
#     new_y = new_y / np.linalg.norm(new_y)
#     rotation = np.linalg.inv(np.stack([new_x, new_y, new_z]))
#
#     if return_rotation_matrix:
#         return pos @ rotation, rotation
#     else:
#         return pos @ rotation


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

        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.separate_data_dir, exist_ok=True)

        if len(os.listdir(self.separate_data_dir)) == 0:
            # data = []
            # for idx, folder in enumerate(folders):
            processed_data_path = osp.join(self.processed_data_dir, f'{self.dataset_name}_data.pth')
            # print(f'Processing data for {folder}: {idx + 1} / {len(folders)}...')
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
            # data.extend(data)

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
                torch.save(graph, osp.join(self.separate_data_dir, f'{i:04d}.pth'))

            sum_num_edges = 0
            sum_num_nodes = 0
            for graph in data:
                sum_num_edges += graph.edge_index.shape[1]
                sum_num_nodes += graph.x.shape[0]
            self.avg_edges = sum_num_edges / sum_num_nodes
            with open(osp.join(self.separate_data_dir, 'metadata.json'), 'w') as f:
                json.dump({'Average edges per node': self.avg_edges}, f)
        else:
            with open(osp.join(self.separate_data_dir, 'metadata.json'), 'r') as f:
                self.avg_edges = json.load(f)

    def get_data(self, dataset_name):
        with open(self.atom_embed, 'r') as f:
            embed = json.load(f)

        with open(self.radii, 'r') as f:
            radii = json.load(f)

        with open(self.ele_label, 'r') as f:
            ele_label = json.load(f)

        data = []
        for file_name in tqdm(sorted(glob.glob(f'{osp.join(self.root, "raw", dataset_name)}/*'))):
            with open(file_name, 'r') as f:
                lines = [line.strip('\n') for line in f.readlines()]

            target = eval(lines[0])
            atom_list = [str(ele_label[atom]) for atom in lines[1].split()]
            pos = torch.tensor([[eval(number) for number in line.split()] for line in lines[2:]])
            x = torch.tensor([embed[atom] for atom in atom_list], dtype=torch.float)

            if self.use_invariance:
                pos = get_invariant_pos(pos)['trans']

            graph = Data(x=x, pos=pos, target=target)
            graph.edge_index = radius_graph(graph.pos, self.cutoff)

            src, dst = graph.edge_index
            displacement = graph.pos[dst] - graph.pos[src]

            distance = (displacement ** 2).sum(dim=1, keepdim=True).sqrt()
            n_st = displacement / distance

            if n_st.isnan().sum() > 0:
                print(f'File {file_name} encountered null!')
                continue

            atom_radii = torch.tensor([radii[atom] for atom in atom_list], dtype=torch.float).reshape(-1, 1)
            a_s = atom_radii[src]
            a_t = atom_radii[dst]
            p_st = torch.cat([distance, distance - a_s, distance - a_t, distance - a_s - a_t], dim=1)

            graph.edge_attr = torch.cat([n_st, p_st / self.cutoff], dim=1)
            data.append(graph)
        torch.save(data, osp.join(self.processed_data_dir, f'{dataset_name}_data.pth'))
        return data

    def __len__(self):
        return len(glob.glob(f'{self.separate_data_dir}/*.pth'))

    def __getitem__(self, idx):
        return torch.load(osp.join(self.separate_data_dir, f'{idx:04d}.pth'))


if __name__ == '__main__':
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
    # val_set = GraphDataset(root='data',
    #                        dataset_name='qm7',
    #                        ele_label='data/ele.json',
    #                        atom_embed='data/atom_init_embedding.json',
    #                        radii='data/radii.json',
    #                        split='val',
    #                        cutoff=6)
    # test_set = GraphDataset(root='data',
    #                         dataset_name='qm7',
    #                         ele_label='data/ele.json',
    #                         atom_embed='data/atom_init_embedding.json',
    #                         radii='data/radii.json',
    #                         split='test',
    #                         cutoff=6)
    pdb.set_trace()
