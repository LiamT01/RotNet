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
from weights import weights

output_dir = f'exp/train_{datetime.now():%Y-%m-%d_%H:%M:%S}'
logger = get_logger(osp.join(output_dir, 'train.log'))

parser = argparse.ArgumentParser()
parser.add_argument('--use_invariance', action='store_true')
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--cutoff', default=4, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', '--bs', default=64, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num_layers', default=5, type=int)
parser.add_argument('--gaussian_num_steps', default=50, type=int)
parser.add_argument('--x_size', default=92, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = GraphDataset(root='data',
                         dataset_name=args.dataset_name,
                         ele_label='data/ele.json',
                         atom_embed='data/atom_init_embedding.json',
                         radii='data/radii.json',
                         split='train',
                         cutoff=args.cutoff,
                         use_invariance=args.use_invariance)
val_set = GraphDataset(root='data',
                       dataset_name=args.dataset_name,
                       ele_label='data/ele.json',
                       atom_embed='data/atom_init_embedding.json',
                       radii='data/radii.json',
                       split='val',
                       cutoff=args.cutoff,
                       use_invariance=args.use_invariance)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)

model = GNN(num_layers=args.num_layers, x_size=args.x_size, hidden_size=args.hidden_size,
            cutoff=args.cutoff, gaussian_num_steps=args.gaussian_num_steps, targets=train_set.metadata['targets'])
model = model.float().to(device)

Loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

logger.info(args)
logger.info(args.dataset_name)
logger.info(model)
logger.info(f'Average edges per node: {train_set.metadata["averageEdgesPerNode"]}')

best_val_loss = float('inf')

for epoch in tqdm(range(1, args.epochs + 1)):
    model.train()
    sum_train_losses = 0
    train_results_per_epoch = []
    for idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        total_loss_per_batch, train_results_per_batch = model(batch, Loss)
        total_loss_per_batch.backward()
        optimizer.step()

        sum_train_losses += total_loss_per_batch.item()
        train_results_per_epoch.append(train_results_per_batch)

        if idx % 20 == 0:
            print(f'Batch {idx + 1}: training loss = {total_loss_per_batch.item()}')

    model.eval()
    sum_val_losses = 0
    val_results_per_epoch = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            total_loss_per_batch, val_results_per_batch = model(batch, Loss)
            sum_val_losses += total_loss_per_batch.item()
            val_results_per_epoch.append(val_results_per_batch)

    train_reduced_losses = reduce_losses(train_results_per_epoch)
    val_reduced_losses = reduce_losses(val_results_per_epoch)

    t = [train_reduced_losses['count'], val_reduced_losses['count']] + \
        [sum_train_losses, sum_val_losses] + \
        [*train_reduced_losses['sum loss'].values()] + \
        [*train_reduced_losses['sum relative loss'].values()] + \
        [*val_reduced_losses['sum loss'].values()] + \
        [*val_reduced_losses['sum relative loss'].values()]
    t = torch.tensor(t, dtype=torch.float, device='cuda')

    num_losses = len(train_set.metadata['targets'])
    t[2] = t[2] / t[0]
    t[3] = t[3] / t[1]
    t[4: 2 * num_losses + 4] = t[4: 2 * num_losses + 4] / t[0]
    t[2 * num_losses + 4:] = t[2 * num_losses + 4:] / t[1]
    t = t[2:].tolist()

    logger.info(f'Epoch {epoch}, Total: train_loss={t[0]:.8f}, val_loss={t[1]:.8f}')

    for name, \
        train_loss, \
        train_relative_loss, \
        val_loss, \
        val_relative_loss in zip([target['name'] for target in train_set.metadata['targets']],
                                 t[2: num_losses + 2],
                                 t[num_losses + 2: 2 * num_losses + 2],
                                 t[2 * num_losses + 2: 3 * num_losses + 2],
                                 t[3 * num_losses + 2:]):
        logger.info(f'\t{name}, '
                    f'Train: loss={train_loss:.8f}, relative_loss={train_relative_loss:.8f}, '
                    f'Val: loss={val_loss:.8f}, relative_loss={val_relative_loss:.8f}')

    if t[1] < best_val_loss:
        logger.info(f'Best val_loss={t[1]} so far was found! Model weights were saved.')
        if epoch < 400:
            torch.save(model.state_dict(), osp.join(output_dir, 'best_weights_early_epoch.pth'))
        else:
            num_digits = int(np.ceil(np.log(args.epochs) / np.log(10) + 1))
            torch.save(model.state_dict(), osp.join(output_dir, f'best_weights_epoch_{epoch:0{num_digits}d}.pth'))

        best_val_loss = t[1]
