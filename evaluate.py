import argparse
import os.path as osp
import pdb
import random
from datetime import datetime
from sys import exit

import numpy as np
import torch
from dataset import GraphDataset
from models import GNN
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import get_logger, reduce_losses

# output_dir = f'exp/train_{datetime.now():%Y-%m-%d_%H:%M:%S}'
# logger = get_logger(osp.join(output_dir, 'train.log'))

parser = argparse.ArgumentParser()
parser.add_argument('--use_invariance', action='store_true')
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--cutoff', default=4, type=float)
parser.add_argument('--batch_size', '--bs', default=64, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_layers', default=5, type=int)
parser.add_argument('--gaussian_num_steps', default=50, type=int)
parser.add_argument('--x_size', default=92, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--checkpoint', required=True, type=str)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_set = GraphDataset(root='data',
                        dataset_name=args.dataset_name,
                        split='test',
                        ele_label='data/ele.json',
                        atom_embed='data/atom_init_embedding.json',
                        radii='data/radii.json',
                        cutoff=args.cutoff,
                        use_invariance=args.use_invariance)

test_loader = DataLoader(test_set, batch_size=args.batch_size)

model = GNN(num_layers=args.num_layers, x_size=args.x_size, hidden_size=args.hidden_size,
            cutoff=args.cutoff, gaussian_num_steps=args.gaussian_num_steps, targets=test_set.metadata['targets'])
model = model.float().to(device)

weights = torch.load(args.checkpoint, map_location=torch.device('cpu'))
model.load_state_dict(weights)

Loss = torch.nn.L1Loss()

# logger.info(args)
# logger.info(model)
# logger.info(f'Average edges per node: {train_set.avg_edges}')

# best_val_loss = float('inf')

model.eval()
sum_test_losses = 0
test_results_per_epoch = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        total_loss_per_batch, test_results_per_batch = model(batch, Loss)
        sum_test_losses += total_loss_per_batch.item()
        test_results_per_epoch.append(test_results_per_batch)

test_reduced_losses = reduce_losses(test_results_per_epoch)

t = [test_reduced_losses['count']] + \
    [sum_test_losses] + \
    [*test_reduced_losses['sum loss'].values()] + \
    [*test_reduced_losses['sum relative loss'].values()]
t = torch.tensor(t, dtype=torch.float, device='cuda')

num_losses = len(test_set.metadata['targets'])
t[1] = t[1] / t[0]
t[2: 2 * num_losses + 2] = t[2: 2 * num_losses + 2] / t[0]
t = t[1:].tolist()

print(f'Total: test_loss={t[0]:.8f}')

for name, \
    test_loss, \
    test_relative_loss in zip([target['name'] for target in test_set.metadata['targets']],
                              t[1: num_losses + 1],
                              t[num_losses + 1:]):
    print(f'\t{name}, '
          f'Test: loss={test_loss:.8f}, relative_loss={test_relative_loss:.8f}')
