import pdb

import numpy as np
import torch
from torch import nn
from torch import tanh, sigmoid
from torch_scatter import scatter_add, scatter_mean
from loss_weights import loss_weights


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)


class GNNLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.f_upd = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
        )
        self.f_int = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.f_mes = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.ind_imp_mask = nn.Parameter(torch.zeros(hidden_size, hidden_size))

    def forward(self, x, edge_index, norm_distance, init_edge_states):
        src, dst = edge_index

        decay_factor = torch.cos(np.pi / 2 * norm_distance).reshape(-1, 1)
        int_imp_mask = decay_factor * self.f_int(torch.cat([x[src], init_edge_states, x[dst]], dim=1))

        m_st = int_imp_mask * self.f_mes(x[src])
        m_t = self.f_mes(x) @ self.ind_imp_mask
        incoming_message = scatter_mean(m_st, index=dst, dim=0)
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
        return self.f_upd(m_t + incoming_message)


class GNN(nn.Module):
    def __init__(self, num_layers, x_size, hidden_size, cutoff, gaussian_num_steps, targets):
        super().__init__()

        self.cutoff = cutoff
        self.gaussian_num_steps = gaussian_num_steps

        self.f_edge = nn.Sequential(
            nn.Linear(4 * gaussian_num_steps, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            GNNLayer(hidden_size=hidden_size) for _ in range(num_layers)
        ])
        self.f_node = nn.Sequential(
            nn.Linear(x_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.targets = targets
        self.f_target = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, target['dim']),
            ) for target in self.targets
        ])

        self.apply(init_weights)

    def forward(self, batch, loss_fn):
        init_edge_states = self.f_edge(
            self.gaussian_expand(batch['ex_norm_displacement'], self.gaussian_num_steps)
        )

        x = self.f_node(batch.x)
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.ex_norm_displacement[:, -1], init_edge_states) + x

        results = []
        total_loss_per_batch = 0
        for target, layer in zip(self.targets, self.f_target):
            if target['level'] == 'node':
                data = layer(x)
            elif target['level'] == 'graph':
                data = layer(scatter_mean(x, index=batch.batch, dim=0))
            else:
                raise Exception(f"Unrecognized level: {target['level']}. It must be either 'graph' or 'node'.")

            ground_truth = batch[target['name']]
            loss = loss_fn(data, ground_truth)
            total_loss_per_batch += loss_weights[target['name']] * loss

            results.append({
                'name': target['name'],
                'level': target['level'],
                'data': data,
                'loss': loss,
                'relative loss': ((data - ground_truth).norm(p=2, dim=1) /
                                  (ground_truth + 1e-9).norm(p=2, dim=1)).mean(),
            })

        return total_loss_per_batch, results

    @staticmethod
    def gaussian_expand(dist, num_steps):
        mu = torch.linspace(0, 1, num_steps).to(dist.device)
        sigma = 1 / (num_steps - 1)
        return torch.exp(-(dist[..., None] - mu) ** 2 / (2 * sigma ** 2)).flatten(start_dim=1)
