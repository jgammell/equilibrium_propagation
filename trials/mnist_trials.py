import os
print(os.getcwd())
os.chdir(os.path.join(os.getcwd(), '..'))
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
from framework import eqp
from framework import datasets
from matplotlib import pyplot as plt
import torch
import numpy as np
import time
import pickle
filepath = os.path.join(os.getcwd(), 'results', 'diabetes_5layer_SW')
import sys
old_stdout = sys.stdout
log_file = open(os.path.join(filepath, 'log.txt'), 'a')
sys.stdout = log_file

topology = \
{
    'layer sizes': [10, 10, 10, 10, 10, 10, 1],
    'network type': 'SW_intra',
    'bypass p': .0756,
    'bypass mag': .579
}
hyperparameters = \
{
    'learning rate': .01,
    'epsilon': .5,
    'beta': 1.0,
    'free iterations': 1000,
    'weakly clamped iterations': 12
}
configuration = \
{
    'batch size': 5,
    'device': 'cuda',
    'seed': 0
}

per_layer_rates = []
correction_matrices = []
training_errors = []
test_errors = []

n_epochs = 100
rate_period = 1
correction_period = 10
Network = eqp.Network(topology, hyperparameters, configuration, datasets.Diabetes)

initial_W = Network.W.clone().cpu().squeeze().numpy()
initial_W_mask = Network.W_mask.clone().cpu().squeeze().numpy()

with open(os.path.join(filepath, 'init.pickle'), 'wb') as F:
    pickle.dump({
            'topology': topology,
            'hyperparameters': hyperparameters,
            'configuration': configuration,
            'dataset': Network.dataset.name,
            'training parameters': {'number of epochs': n_epochs, 'rate period': rate_period, 'correction_period': correction_period},
            'initial weight': initial_W,
            'initial mask': initial_W_mask}, F)

for epoch_idx in np.arange(n_epochs):
    print('Starting epoch %d.'%(epoch_idx+1))
    t0 = time.time()
    Network.train_epoch()
    Network.calculate_test_error()
    training_errors.append(Network.training_error)
    test_errors.append(Network.test_error)
    if epoch_idx % rate_period == 0:
        per_layer_rates.append([])
        for conn in Network.interlayer_connections:
            correction = torch.norm((Network.dW*conn)/torch.sqrt(torch.norm(conn, p=1)))
            per_layer_rates[-1].append(float(correction.cpu()))
    if epoch_idx % correction_period == 0:
        correction_matrices.append(Network.dW.clone().cpu().squeeze().numpy())
    print('\tDone.')
    print('\tTime taken:', (time.time()-t0))
    print('\tTraining error:', Network.training_error)
    print('\tTest error:', Network.test_error)
    with open(os.path.join(filepath, 'e%d.pickle'%(epoch_idx)), 'wb') as F:
        pickle.dump({
                'training error': Network.training_error,
                'test error': Network.test_error,
                'per-layer rates': per_layer_rates[-1] if (epoch_idx % rate_period == 0) else None,
                'correction matrix': correction_matrices[-1] if (epoch_idx % correction_period == 0) else None}, F)

with open(os.path.join(filepath, 'final_network.pickle'), 'wb') as F:
    pickle.dump(Network, F)

