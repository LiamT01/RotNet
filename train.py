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
    train_losses = []
    relative_train_losses = []
    for idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        loss = Loss(out, batch.target.float())
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

        relative_train_loss = ((out - batch.target) / (batch.target + 1e-9)).abs().mean()
        relative_train_losses.append(relative_train_loss)
        if idx % 20 == 0:
            print(f'Batch {idx + 1}: training loss = {loss.item()}')

    model.eval()
    val_losses = []
    relative_val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
            loss = Loss(out, batch.target.float())
            val_losses.append(loss)
            relative_val_loss = ((out - batch.target) / (batch.target + 1e-9)).abs().mean()
            relative_val_losses.append(relative_val_loss)
    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)
    relative_train_loss = sum(relative_train_losses) / len(relative_train_losses)
    relative_val_loss = sum(relative_val_losses) / len(relative_val_losses)

    # if epoch == 1 or epoch % 10 == 0:
    logger.info(f'Epoch {epoch}: train_loss={train_loss:.8f}, '
                f'relative_train_loss={relative_train_loss:.8f}, '
                f'val_loss={val_loss:.8f}, '
                f'relative_val_loss={relative_val_loss:.8f}')

    if val_loss < best_val_loss:
        logger.info(f'Best val_loss={val_loss} so far was found! Model weights were saved.')
        if epoch < 400:
            torch.save(model.state_dict(), osp.join(output_dir, 'best_weights_early_epoch.pth'))
        else:
            num_digits = int(np.ceil(np.log(args.epochs) / np.log(10) + 1))
            torch.save(model.state_dict(), osp.join(output_dir, f'best_weights_epoch_{epoch:0{num_digits}d}.pth'))

        best_val_loss = val_loss
