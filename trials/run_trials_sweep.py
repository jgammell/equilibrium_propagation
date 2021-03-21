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
import sys
old_stdout = sys.stdout

default_topology_mlff = {
        'network type': 'MLFF',
        'bypass p': None,
        'bypass mag': None}
default_topology_sw = {
        'network type': 'SW_intra',
        'bypass p': .0756}
default_hyperparameters = {
        'epsilon': .5,
        'beta': 1.0}
default_configuration = {
        'device': 'cuda',
        'seed': 0}

mnist_1layer_mlff_topology = default_topology_mlff.copy()
mnist_1layer_mlff_topology['layer sizes'] = [28**2, 500, 10]
mnist_1layer_sw_topology = default_topology_sw.copy()
mnist_1layer_sw_topology['layer sizes'] = [28**2, 500, 10]
mnist_1layer_sw_topology['bypass mag'] = .0884
mnist_1layer_sw_topology['bypass p'] = .005
mnist_1layer_hyperparameters = default_hyperparameters.copy()
mnist_1layer_hyperparameters['learning rate'] = .05
mnist_1layer_hyperparameters['free iterations'] = 20
mnist_1layer_hyperparameters['weakly clamped iterations'] = 4
mnist_1layer_configuration = default_configuration.copy()
mnist_1layer_configuration['batch size'] = 20

fmnist_1layer_mlff_topology = default_topology_mlff.copy()
fmnist_1layer_mlff_topology['layer sizes'] = [28**2, 500, 10]
fmnist_1layer_sw_topology = default_topology_sw.copy()
fmnist_1layer_sw_topology['layer sizes'] = [28**2, 500, 10]
fmnist_1layer_sw_topology['bypass mag'] = .0884
fmnist_1layer_sw_topology['bypass p'] = .005
fmnist_1layer_hyperparameters = default_hyperparameters.copy()
fmnist_1layer_hyperparameters['learning rate'] = .05
fmnist_1layer_hyperparameters['free iterations'] = 20
fmnist_1layer_hyperparameters['weakly clamped iterations'] = 4
fmnist_1layer_configuration = default_configuration.copy()
fmnist_1layer_configuration['batch size'] = 20

diabetes_1layer_mlff_topology = default_topology_mlff.copy()
diabetes_1layer_mlff_topology['layer sizes'] = [10, 50, 1]
diabetes_1layer_sw_topology = default_topology_sw.copy()
diabetes_1layer_sw_topology['layer sizes'] = [10, 50, 1]
diabetes_1layer_sw_topology['bypass mag'] = .329
diabetes_1layer_sw_topology['bypass p'] = .003
diabetes_1layer_hyperparameters = default_hyperparameters.copy()
diabetes_1layer_hyperparameters['learning rate'] = .05
diabetes_1layer_hyperparameters['free iterations'] = 20
diabetes_1layer_hyperparameters['weakly clamped iterations'] = 4
diabetes_1layer_configuration = default_configuration.copy()
diabetes_1layer_configuration['batch size'] = 1

wine_1layer_mlff_topology = default_topology_mlff.copy()
wine_1layer_mlff_topology['layer sizes'] = [13, 50, 3]
wine_1layer_sw_topology = default_topology_sw.copy()
wine_1layer_sw_topology['layer sizes'] = [13, 50, 3]
wine_1layer_sw_topology['bypass mag'] = .322
wine_1layer_sw_topology['bypass p'] = .006
wine_1layer_hyperparameters = default_hyperparameters.copy()
wine_1layer_hyperparameters['learning rate'] = .05
wine_1layer_hyperparameters['free iterations'] = 20
wine_1layer_hyperparameters['weakly clamped iterations'] = 4
wine_1layer_configuration = default_configuration.copy()
wine_1layer_configuration['batch size'] = 1

mnist_1layer_mlff = {
        'folder': 'mnist_1layer_mlff',
        'number of epochs': 100,
        'topology': mnist_1layer_mlff_topology,
        'hyperparameters': mnist_1layer_hyperparameters,
        'configuration': mnist_1layer_configuration,
        'dataset': datasets.MNIST}
mnist_1layer_sw = {
        'folder': 'mnist_1layer_sw',
        'number of epochs': 100,
        'topology': mnist_1layer_sw_topology,
        'hyperparameters': mnist_1layer_hyperparameters,
        'configuration': mnist_1layer_configuration,
        'dataset': datasets.MNIST}
fmnist_1layer_mlff = {
        'folder': 'fmnist_1layer_mlff',
        'number of epochs': 100,
        'topology': fmnist_1layer_mlff_topology,
        'hyperparameters': fmnist_1layer_hyperparameters,
        'configuration': fmnist_1layer_configuration,
        'dataset': datasets.FashionMNIST}
fmnist_1layer_sw = {
        'folder': 'fmnist_1layer_sw',
        'number of epochs': 100,
        'topology': fmnist_1layer_sw_topology,
        'hyperparameters': fmnist_1layer_hyperparameters,
        'configuration': fmnist_1layer_configuration,
        'dataset': datasets.FashionMNIST}
diabetes_1layer_mlff = {
        'folder': 'diabetes_1layer_mlff',
        'number of epochs': 100,
        'topology': diabetes_1layer_mlff_topology,
        'hyperparameters': diabetes_1layer_hyperparameters,
        'configuration': diabetes_1layer_configuration,
        'dataset': datasets.Diabetes}
diabetes_1layer_sw = {
        'folder': 'diabetes_1layer_sw',
        'number of epochs': 100,
        'topology': diabetes_1layer_sw_topology,
        'hyperparameters': diabetes_1layer_hyperparameters,
        'configuration': diabetes_1layer_configuration,
        'dataset': datasets.Diabetes}
wine_1layer_mlff = {
        'folder': 'wine_1layer_mlff',
        'number of epochs': 100,
        'topology': wine_1layer_mlff_topology,
        'hyperparameters': wine_1layer_hyperparameters,
        'configuration': wine_1layer_configuration,
        'dataset': datasets.Wine}
wine_1layer_sw = {
        'folder': 'wine_1layer_sw',
        'number of epochs': 100,
        'topology': wine_1layer_sw_topology,
        'hyperparameters': wine_1layer_hyperparameters,
        'configuration': wine_1layer_configuration,
        'dataset': datasets.Wine}

mnist_3layer_mlff = {
'folder': 'mnist_3layer_mlff',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 500, 500, 500, 10],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .02,
'epsilon': .5,
'beta': 1.0,
'free iterations': 500,
'weakly clamped iterations': 8},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.MNIST}
mnist_3layer_sw = {
'folder': 'mnist_3layer_sw',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 500, 500, 500, 10],
'network type': 'SW_intra',
'bypass p': .1,
'bypass mag': .0829},
'hyperparameters': {
'learning rate': .02,
'epsilon': .5,
'beta': 1.0,
'free iterations': 500,
'weakly clamped iterations': 8},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.MNIST}
mnist_3layer_swni = {
'folder': 'mnist_3layer_swni',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 500, 500, 500, 10],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .0829},
'hyperparameters': {
'learning rate': .02,
'epsilon': .5,
'beta': 1.0,
'free iterations': 500,
'weakly clamped iterations': 8},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.MNIST}

mnist_5layer_swni = {
'folder': 'mnist_5layer_swni',
'number of epochs': 10,
'topology': {
'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .168},
'hyperparameters': {
'learning rate': .015,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.MNIST}
mnist_5layer_sw = {
'folder': 'mnist_5layer_sw',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
'network type': 'SW_intra',
'bypass p': .1,
'bypass mag': .168},
'hyperparameters': {
'learning rate': .015,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.MNIST}
fmnist_5layer_swni = {
'folder': 'fmnist_5layer_swni',
'number of epochs': 10,
'topology': {
'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .168},
'hyperparameters': {
'learning rate': .015,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.FashionMNIST}
fmnist_5layer_sw = {
'folder': 'fmnist_5layer_sw',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
'network type': 'SW_intra',
'bypass p': .1,
'bypass mag': .168},
'hyperparameters': {
'learning rate': .015,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.FashionMNIST}
fmnist_5layer_mlff = {
'folder': 'fmnist_5layer_mlff',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .015,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.FashionMNIST}
mnist_5layer_mlff = {
'folder': 'mnist_5layer_mlff',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 100, 100, 100, 100, 100, 10],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .015,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 20,
'device': 'cuda',
'seed': 0},
'dataset': datasets.MNIST}

trial = mnist_5layer_swni

for p in [0] + [pp for pp in np.logspace(-4, 0, 30)]:
    filepath = os.path.join(os.getcwd(), 'results', 'mnist_5layer_sweep', 'trial_p%.06f'%(p))
    os.mkdir(filepath)
    log_file = open(os.path.join(filepath, 'log.txt'), 'a')
    sys.stdout = log_file
    trial['topology']['bypass p'] = float(p)

    per_layer_rates = []
    correction_matrices = []
    training_errors = []
    test_errors = []
    states = []
    n_epochs = trial['number of epochs']
    Network = eqp.Network(trial['topology'], trial['hyperparameters'], trial['configuration'], trial['dataset'])
    initial_W = Network.W.clone().cpu().squeeze().numpy()
    initial_W_mask = Network.W_mask.clone().cpu().squeeze().numpy()

    with open(os.path.join(filepath, 'init.pickle'), 'wb') as F:
        pickle.dump({
            'trial settings': trial,
            'initial weight': initial_W,
            'initial mask': initial_W_mask}, F)
            
    Network.calculate_test_error()
    training_errors.append(Network.test_error)
    test_errors.append(Network.test_error)
    mean_dW = torch.zeros(Network.W.shape).to(trial['configuration']['device'])
    for epoch_idx in np.arange(n_epochs):
        print('Starting epoch %d.'%(epoch_idx+1))
        t0 = time.time()
        Network.train_epoch()
        #Network.calculate_training_error()
        Network.calculate_test_error()
        states.append((int(torch.count_nonzero(Network.s==0).cpu()), int(torch.count_nonzero(Network.s==1).cpu())))
        training_errors.append(Network.training_error)
        test_errors.append(Network.test_error)
        mean_dW = torch.abs(Network.mean_dW)/(epoch_idx+1) + (epoch_idx/(epoch_idx+1))*mean_dW
        for p in Network.per_layer_rates:
            per_layer_rates.append(p)
        print('\tDone.')
        print('\tTime taken:', (time.time()-t0))
        print('\tTraining error:', Network.training_error)
        print('\tTest error:', Network.test_error)
        with open(os.path.join(filepath, 'e%d.pickle'%(epoch_idx)), 'wb') as F:
            pickle.dump({
                'training error': Network.training_error,
                'test error': Network.test_error,
                #'true training error': Network.true_training_error,
                'per-layer rates': per_layer_rates[-Network.dataset.n_trainb:],
                'mean dW': mean_dW.clone().squeeze().cpu().numpy(),
                'states': states[-1]}, F)
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
        rates = [trial['hyperparameters']['learning rate']*p[idx] for p in per_layer_rates]
        plt.plot(np.arange(len(rates))+1, rates, '.')
    plt.xlabel('Epoch')
    plt.ylabel('Magnitude of correction')
    plt.yscale('log')
    plt.savefig(os.path.join(filepath, 'rates.png'))
    plt.imsave(os.path.join(filepath, 'w.png'), Network.W.clone().squeeze().cpu().numpy())
    plt.imsave(os.path.join(filepath, 'wmask.png'), Network.W_mask.clone().squeeze().cpu().numpy())
    plt.imsave(os.path.join(filepath, 'mean_dW.png'), mean_dW.clone().squeeze().cpu().numpy())
    
    
    
    
    
    
    
    
