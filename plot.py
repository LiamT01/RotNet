import argparse
import os.path as osp
import pdb

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)

args = parser.parse_args()

train_losses = []
val_losses = []

with open(osp.join(args.exp, 'train.log'), 'r') as f:
    for line in f:
        if 'Epoch' in line:
            train_losses.append(float(line.split()[-2].split('=')[-1].strip(',')))
            val_losses.append(float(line.split()[-1].split('=')[-1]))

plt.ylim([0, 60])
plt.plot(train_losses[100:5000], label='Train', color='blue')
plt.plot(val_losses[100:5000], label='Val', color='red')
plt.legend()
plt.savefig(osp.join(args.exp, 'plot.png'))
