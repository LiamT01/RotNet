import argparse
import os.path as osp
import pdb
import numpy as np

import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser()
# parser.add_argument('--exp', type=str, required=True)

# args = parser.parse_args()

train_losses = []
val_losses = []

for exp, label in zip(['exp/train_2022-09-14_07:51:39', 'exp/train_2022-09-14_07:50:25'],
                      ['Invariant', 'Default']):
    epochs = []
    band_gap_rel_losses = []
    with open(osp.join(exp, 'train.log'), 'r') as f:
        for i, line in enumerate(f):
            # if 'Epoch' in line:
            #     train_losses.append(float(line.split()[-2].split('=')[-1].strip(',')))
            #     val_losses.append(float(line.split()[-1].split('=')[-1]))
            if 'Epoch' in line:
                epoch = int(line.split(',')[0].split(' ')[-1])
            if "band_gap, Train:" in line:
                value = float(line.split(',')[-1].split('relative_loss=')[-1].replace('\n', ''))
                # if epoch <= 2000 and not (label == 'Invariant' and value > 10 ** 10):
                if value <= 2 and epoch <= 2000:
                    epochs.append(epoch)
                    band_gap_rel_losses.append(value)

    # plt.ylim([0, 60])
    # plt.plot(train_losses[100:5000], label='Train', color='blue')
    # plt.plot(val_losses[100:5000], label='Val', color='red')
    # plt.legend()
    # plt.savefig(osp.join(args.exp, 'plot.png'))

    # band_gap_rel_losses = np.log10(np.array(band_gap_rel_losses))
    plt.plot(epochs, band_gap_rel_losses, label=label, alpha=0.7)
plt.legend()
plt.title('Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Relative loss')
plt.savefig(osp.join('band_gap_val_rel_loss_filtered.png'))
