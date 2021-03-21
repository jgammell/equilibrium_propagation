import os
os.chdir(os.path.join(os.getcwd(), '..'))
from framework import eqp
from framework import datasets
from matplotlib import pyplot as plt
import torch
import numpy as np

topology = \
{
    'layer sizes': [10, 5, 5, 5, 5, 5, 1],
    'network type': 'SW_intra',
    'bypass p': .1,
    'bypass mag': .05
}
hyperparameters = \
{
    'learning rate': .02,
    'epsilon': .03,
    'beta': .04,
    'free iterations': 100,
    'weakly clamped iterations': 20
}
configuration = \
{
    'batch size': 20,
    'device': 'cpu',
    'seed': 0
}

Network = eqp.Network(topology, hyperparameters, configuration, datasets.Diabetes)
(fig, ax) = plt.subplots(1, 2)
ax[0].imshow(np.abs(Network.W.squeeze().numpy()), vmin=0, vmax=.1*torch.max(torch.abs(Network.W)))
ax[1].imshow(Network.W_mask.squeeze().numpy(), vmin=0, vmax=1)
fig.savefig(os.path.join(os.getcwd(), 'results', 'validate_code_figures', 'weight_matrices.jpg'), quality=100)
(fig, ax) = plt.subplots(1, len(Network.interlayer_connections))
for i, conn in zip(range(len(Network.interlayer_connections)), Network.interlayer_connections):
    ax[i].imshow(conn.squeeze().numpy(), vmin=0, vmax=1)
fig.savefig(os.path.join(os.getcwd(), 'results', 'validate_code_figures', 'connection_masks.jpg'), quality=100)
for i in range(5):
    Network.train_epoch()
    Network.calculate_test_error()
    print('Training error:', Network.training_error)
    print('Test error:', Network.test_error)

