#%%
# -*- coding: utf-8 -*-
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
filepath = os.path.join(os.getcwd(), 'results', 'diabetes_5layer_swni')
log_file = open(os.path.join(filepath, 'log.txt'), 'w')
old_output = sys.stdout
sys.stdout = log_file

topology = \
{
    'layer sizes': [10, 10, 10, 10, 10, 10, 1],
    'network type': 'SW_no_intra',
    'bypass p': .0756,
    'bypass mag': .568
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


"""errors = []
for lr in np.linspace(.001, .1, 100):
    hyperparameters['learning rate'] = float(lr)
    Network = eqp.Network(topology, hyperparameters, configuration, datasets.Wine)
    Network.train_epoch()
    errors.append(Network.training_error)
    print('rate', lr, 'error', errors[-1], file=old_output)
plt.plot(np.arange(.001, .1, 100), errors, '.')
plt.savefig(os.path.join(filepath, 'lr_sweep.png'))
assert False""";

with open(os.path.join(filepath, 'init.pickle'), 'wb') as F:
    pickle.dump({
        'topology': topology,
        'hyperparameters': hyperparameters,
        'configuration': configuration}, F)
        

per_layer_rates = []
training_errors = []
test_errors = []

n_epochs = 10
Network = eqp.Network(topology, hyperparameters, configuration, datasets.Diabetes)
Network.calculate_test_error()
training_errors.append(Network.test_error)
test_errors.append(Network.test_error)
for epoch_idx in range(n_epochs):
    print('Starting epoch %d.'%(epoch_idx+1))
    t0 = time.time()
    Network.train_epoch()
    Network.calculate_test_error()
    training_errors.append(Network.training_error)
    test_errors.append(Network.test_error)
    per_layer_rates.append([])
    for i in range(len(Network.interlayer_connections)):
    	per_layer_rates[-1].append(np.mean([p[i] for p in Network.per_layer_rates]))
    with open(os.path.join(filepath, 'e%d.pickle'%(epoch_idx)), 'wb') as F:
        pickle.dump({
            'training error': training_errors[-1],
            'test error': test_errors[-1],
            'per-layer rates': per_layer_rates[-1]}, F)
    print('\tDone.')
    print('\tTime taken:', (time.time()-t0))
    print('\tTraining error:', training_errors[-1])
    print('\tTest error:', test_errors[-1])

with open(os.path.join(filepath, 'final_network.pickle'), 'wb') as F:
    pickle.dump(Network, F)

plt.figure()
plt.plot(np.arange(len(training_errors)), training_errors, color='blue')
plt.plot(np.arange(len(test_errors)), test_errors, '--', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.savefig(os.path.join(filepath, 'error.png'))

plt.figure()
for idx in range(len(per_layer_rates[0])):
    rates = [hyperparameters['learning rate']*p[idx] for p in per_layer_rates]
    plt.plot(np.arange(len(rates))+1, rates, '.')
plt.xlabel('Epoch')
plt.ylabel('Magnitude of correction')
plt.yscale('log')
plt.savefig(os.path.join(filepath, 'rates.png'))

plt.imsave(os.path.join(filepath, 'w.png'), Network.W.clone().squeeze().cpu().numpy())
plt.imsave(os.path.join(filepath, 'wmask.png'), Network.W_mask.clone().squeeze().cpu().numpy())
