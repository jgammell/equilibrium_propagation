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
filepath = os.path.join(os.getcwd(), 'results', 'diabetes_findlr')
import sys
#old_stdout = sys.stdout
#log_file = open(os.path.join(filepath, 'log.txt'), 'a')
#sys.stdout = log_file

topology = \
{
    'layer sizes': [10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1],
    'network type': 'MLFF',
    'bypass p': None,
    'bypass mag': None
}
hyperparameters = \
{
    'learning rate': None,
    'epsilon': .5,
    'beta': 1.0,
    'free iterations': 200,
    'weakly clamped iterations': 22
}
configuration = \
{
    'batch size': 5,
    'device': 'cuda',
    'seed': 0
}

t0 = time.time()
training_errors = []
for learning_rate in np.arange(.001, 1, .005):
    hyperparameters['learning rate'] = float(learning_rate)
    Network = eqp.Network(topology, hyperparameters, configuration, datasets.Diabetes)
    Network.train_epoch()
    training_errors.append((learning_rate, Network.training_error))
print('Seconds taken: %.03f'%(time.time()-t0))
print('%.03f seconds per trial.'%((time.time()-t0)/len(training_errors)))
plt.plot([val[0] for val in training_errors], [val[1] for val in training_errors], '.')

with open(os.path.join(filepath, '10layer.pickle'), 'wb') as F:
    pickle.dump(training_errors, F)