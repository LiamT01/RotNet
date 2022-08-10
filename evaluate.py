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
from utils import get_logger

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
            cutoff=args.cutoff, gaussian_num_steps=args.gaussian_num_steps)
model = model.float().to(device)

weights = torch.load(args.checkpoint, map_location=torch.device('cpu'))
# torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(weights, "module.")
model.load_state_dict(weights)

Loss = torch.nn.L1Loss()

# logger.info(args)
# logger.info(model)
# logger.info(f'Average edges per node: {train_set.avg_edges}')

# best_val_loss = float('inf')

model.eval()
test_losses = []
relative_test_losses = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch).flatten()
        loss = Loss(out, batch.target.float())
        test_losses.append(loss)
        relative_test_loss = ((out - batch.target) / (batch.target + 1e-9)).abs().mean()
        relative_test_losses.append(relative_test_loss)

test_loss = sum(test_losses) / len(test_losses)
relative_test_loss = sum(relative_test_losses) / len(relative_test_losses)

# logger.info(f'test_loss={test_loss:.8f}, '
#             f'relative_test_loss={relative_test_loss:.8f}')
print(f'test_loss={test_loss:.8f}, '
      f'relative_test_loss={relative_test_loss:.8f}')
