import pdb

import numpy as np
import torch
from torch import nn
from torch import tanh, sigmoid
from torch_scatter import scatter_add, scatter_mean
from weights import weights
from transforms import gaussian_expand


class GNNLayer(nn.Module):
    def __init__(self, hidden_size, cutoff, f_b, gaussian_num_steps):
        super().__init__()

        self.hidden_size = hidden_size
        self.cutoff = cutoff
        self.gaussian_num_steps = gaussian_num_steps

        self.f_n = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.f_e = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.f_b = f_b
        self.f_d = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.ones(hidden_size).reshape(1, -1))

    def forward(self, x, edge_attr, edge_index):
        src, dst = edge_index

        gaussian_transformed = self.f_b(gaussian_expand(edge_attr, self.gaussian_num_steps))
        coefficient = torch.cos(np.pi / 2 * edge_attr[:, 3]).reshape(-1, 1)
        cond_filter = coefficient * self.f_e(torch.cat([x[src], gaussian_transformed, x[dst]], dim=1))

        m_st = cond_filter * self.f_d(x[src])
        m_t = self.v * self.f_d(x)
        incoming_message = scatter_add(m_st, index=dst, dim=0)
        incoming_message = torch.cat(
            [
                incoming_message,
                torch.zeros(
                    x.shape[0] - incoming_message.shape[0],
                    x.shape[1]
                ).to(incoming_message.device)
            ],
            dim=0
        )
        return self.f_n(m_t + incoming_message) + x


class GNN(nn.Module):
    def __init__(self, num_layers, x_size, hidden_size, cutoff, gaussian_num_steps, targets):
        super().__init__()

        self.f_b = nn.Linear(7 * gaussian_num_steps, hidden_size)
        self.layers = nn.ModuleList([
            GNNLayer(
                hidden_size=hidden_size,
                cutoff=cutoff,
                f_b=self.f_b,
                gaussian_num_steps=gaussian_num_steps
            ) for _ in range(num_layers)
        ])
        self.f_x = nn.Linear(x_size, hidden_size)
        self.targets = targets
        self.f_target = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_size // 2),
                nn.Linear(hidden_size // 2, target['dim']),
            ) for target in self.targets
        ])

    def forward(self, batch, loss_fn):
        batch.x = self.f_x(batch.x)
        for layer in self.layers:
            batch.x = layer(batch.x, batch.edge_attr, batch.edge_index)

        results = []
        total_loss_per_batch = 0
        for target, layer in zip(self.targets, self.f_target):
            if target['level'] == 'node':
                data = layer(batch.x)
            elif target['level'] == 'graph':
                data = layer(scatter_mean(batch.x, index=batch.batch, dim=0))
            else:
                raise Exception(f"Unrecognized level: {target['level']}. It must be either 'graph' or 'node'.")

            name = target['name']
            ground_truth = batch[name]
            loss = loss_fn(data, ground_truth)
            total_loss_per_batch += weights[name] * loss

            results.append({
                'name': name,
                'level': target['level'],
                'data': data,
                'loss': loss,
                'relative loss': ((data - ground_truth).norm(p=2, dim=1) /
                                  (ground_truth + 1e-9).norm(p=2, dim=1)).mean(),
            })

        return total_loss_per_batch, results
