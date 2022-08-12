import argparse
import os
import os.path as osp
import pdb
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset import GraphDataset
from models import GNN
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import get_logger, reduce_losses, get_num_digits
from weights import weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true')
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

    #########################################################
    parser.add_argument('--num_nodes', type=int, default=1, metavar='N')
    parser.add_argument('--num_gpus_per_node', type=int, default=1, help='Number of gpus per node.')
    parser.add_argument('--node_rank', type=int, default=0, help='Ranking within the nodes.')
    parser.add_argument('--master_addr', type=str, default='0.0.0.0')
    parser.add_argument('--master_port', type=str, default='8888')
    #########################################################

    args = parser.parse_args()

    if args.distributed:
        ############################################################
        args.world_size = args.num_gpus_per_node * args.num_nodes  #
        os.environ['MASTER_ADDR'] = args.master_addr               #
        os.environ['MASTER_PORT'] = args.master_port               #
        mp.spawn(train, nprocs=args.world_size, args=(args,))      #
        ############################################################
    else:
        train(0, args)


def train(gpu, args):
    rank = -1
    if args.distributed:
        ############################################################
        rank = args.node_rank * args.num_gpus_per_node + gpu
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )
        ############################################################

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank <= 0:
        output_dir = f'exp/train_{datetime.now():%Y-%m-%d_%H:%M:%S}'
        logger = get_logger(osp.join(output_dir, 'train.log'))

    torch.cuda.set_device(gpu)

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
    if args.distributed:
        ################################################################
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=args.world_size,
            rank=rank
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set,
            num_replicas=args.world_size,
            rank=rank
        )
        ################################################################

    if args.distributed:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            ##############################
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            sampler=train_sampler,
            ##############################
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            ##############################
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            sampler=val_sampler,
            ##############################
        )
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = GNN(num_layers=args.num_layers, x_size=args.x_size, hidden_size=args.hidden_size,
                cutoff=args.cutoff, gaussian_num_steps=args.gaussian_num_steps, targets=train_set.metadata['targets'])
    model = model.float().cuda(gpu)

    if args.distributed:
        ###############################################################
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        ###############################################################

    loss_fn = torch.nn.L1Loss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if rank <= 0:
        logger.info(args)
        logger.info(args.dataset_name)
        logger.info(weights)
        logger.info(model)
        logger.info(f'Average edges per node: {train_set.metadata["averageEdgesPerNode"]}')

    best_val_loss = float('inf')

    for epoch in tqdm(range(1, args.epochs + 1)):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        sum_train_losses = 0
        train_results_per_epoch = []

        for idx, batch in enumerate(train_loader):
            batch = batch.cuda(non_blocking=True)
            optimizer.zero_grad()
            total_loss_per_batch, train_results_per_batch = model(batch, loss_fn)
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
                batch = batch.cuda(non_blocking=True)
                total_loss_per_batch, val_results_per_batch = model(batch, loss_fn)
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

        if args.distributed:
            dist.barrier()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        num_losses = len(train_set.metadata['targets'])
        t[2] = t[2] / t[0]
        t[3] = t[3] / t[1]
        t[4: 2 * num_losses + 4] = t[4: 2 * num_losses + 4] / t[0]
        t[2 * num_losses + 4:] = t[2 * num_losses + 4:] / t[1]
        t = t[2:].tolist()

        if rank <= 0:
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
                logger.info(f'\t\tBest val_loss={t[1]} so far was found! Model weights were saved.')
                if epoch < 400:
                    if args.distributed:
                        torch.save(model.module.state_dict(), osp.join(output_dir, 'best_weights_early_epoch.pth'))
                    else:
                        torch.save(model.state_dict(), osp.join(output_dir, 'best_weights_early_epoch.pth'))
                else:
                    num_digits = get_num_digits(args.epochs)
                    if args.distributed:
                        torch.save(model.module.state_dict(),
                                   osp.join(output_dir, f'best_weights_epoch_{epoch:0{num_digits}d}.pth'))
                    else:
                        torch.save(model.state_dict(),
                                   osp.join(output_dir, f'best_weights_epoch_{epoch:0{num_digits}d}.pth'))

                best_val_loss = t[1]


if __name__ == '__main__':
    main()
