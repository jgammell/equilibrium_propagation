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
filepath = os.path.join(os.getcwd(), 'results', 'mnist_findlr')
import sys
#old_stdout = sys.stdout
#log_file = open(os.path.join(filepath, 'log.txt'), 'a')
#sys.stdout = log_file

topology = \
{
    'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
    'network type': 'MLFF',
    'bypass p': None,
    'bypass mag': None
}
hyperparameters = \
{
    'learning rate': None,
    'epsilon': .5,
    'beta': 1.0,
    'free iterations': 1000,
    'weakly clamped iterations': 12
}
configuration = \
{
    'batch size': 20,
    'device': 'cuda',
    'seed': 0
}

t0 = time.time()
training_errors = []
for learning_rate in np.arange(.003, .03, .003):
    hyperparameters['learning rate'] = float(learning_rate)
    Network = eqp.Network(topology, hyperparameters, configuration, datasets.MNIST)
    Network.train_epoch()
    Network.train_epoch()
    training_errors.append((learning_rate, Network.training_error))
    print('Learning rate: %f, Training error: %f.'%(learning_rate, Network.training_error))
print('Seconds taken: %.03f'%(time.time()-t0))
print('%.03f seconds per trial.'%((time.time()-t0)/len(training_errors)))
with open(os.path.join(filepath, '5layer.pickle'), 'wb') as F:
    pickle.dump(training_errors, F)
plt.plot([val[0] for val in training_errors], [val[1] for val in training_errors], '.')

