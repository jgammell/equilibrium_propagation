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

diabetes_5layer_mlff = {
'folder': 'diabetes_5layer_mlff',
'number of epochs': 10,
'topology': {
'layer sizes': [10, 10, 10, 10, 10, 10, 1],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Diabetes}
diabetes_5layer_swni = {
'folder': 'diabetes_5layer_swni',
'number of epochs': 10,
'topology': {
'layer sizes': [10, 10, 10, 10, 10, 10, 1],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .579},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Diabetes}
diabetes_8layer_mlff = {
'folder': 'diabetes_8layer_mlff',
'number of epochs': 10,
'topology': {
'layer sizes': [10, 10, 10, 10, 10, 10, 10, 10, 10, 1],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 5000,
'weakly clamped iterations': 18},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Diabetes}
diabetes_8layer_swni = {
'folder': 'diabetes_8layer_swni',
'number of epochs': 10,
'topology': {
'layer sizes': [10, 10, 10, 10, 10, 10, 10, 10, 10, 1],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .569},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 5000,
'weakly clamped iterations': 18},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Diabetes}
wine_5layer_mlff = {
'folder': 'wine_5layer_mlff',
'number of epochs': 10,
'topology': {
'layer sizes': [13, 10, 10, 10, 10, 10, 3],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Wine}
wine_5layer_swni = {
'folder': 'wine_5layer_swni',
'number of epochs': 10,
'topology': {
'layer sizes': [13, 10, 10, 10, 10, 10, 3],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .563},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 1000,
'weakly clamped iterations': 12},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Wine}
wine_8layer_mlff = {
'folder': 'wine_8layer_mlff',
'number of epochs': 10,
'topology': {
'layer sizes': [13, 10, 10, 10, 10, 10, 10, 10, 10, 3],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 5000,
'weakly clamped iterations': 18},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
'dataset': datasets.Wine}
wine_8layer_swni = {
'folder': 'wine_8layer_swni',
'number of epochs': 10,
'topology': {
'layer sizes': [13, 10, 10, 10, 10, 10, 10, 10, 10, 3],
'network type': 'SW_no_intra',
'bypass p': .1,
'bypass mag': .558},
'hyperparameters': {
'learning rate': .01,
'epsilon': .5,
'beta': 1.0,
'free iterations': 5000,
'weakly clamped iterations': 18},
'configuration': {
'batch size': 5,
'device': 'cuda',
'seed': 0},
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
mnist_3layer_mlffpl = {
'folder': 'mnist_3layer_mlffpl',
'number of epochs': 100,
'topology': {
'layer sizes': [28**2, 500, 500, 500, 10],
'network type': 'MLFF',
'bypass p': None,
'bypass mag': None},
'hyperparameters': {
'learning rate': [.128, .032, .008, .002],
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
'number of epochs': 200,
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
mnist_5layer_swni = {
'folder': 'mnist_5layer_swni',
'number of epochs': 100,
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
fmnist_5layer_swni = {
'folder': 'fmnist_5layer_swni',
'number of epochs': 100,
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


trials = [wine_8layer_mlff, wine_8layer_swni, diabetes_8layer_mlff, diabetes_8layer_swni]

for trial in trials:
    filepath = os.path.join(os.getcwd(), 'results', trial['folder'])
    log_file = open(os.path.join(filepath, 'log.txt'), 'a')
    sys.stdout = log_file

    per_layer_rates = []
    correction_matrices = []
    training_errors = []
    test_errors = []
    states = []
    n_epochs = trial['number of epochs']
    Network = eqp.Network(trial['topology'], trial['hyperparameters'], trial['configuration'], trial['dataset'])
    Network.calculate_test_error()
    training_errors.append(Network.test_error)
    test_errors.append(Network.test_error)
    epochs = np.arange(n_epochs)
    initial_W = Network.W.clone().cpu().squeeze().numpy()
    initial_W_mask = Network.W_mask.clone().cpu().squeeze().numpy()

    with open(os.path.join(filepath, 'init.pickle'), 'wb') as F:
        pickle.dump({
            'trial settings': trial,
            'initial weight': initial_W,
            'initial mask': initial_W_mask}, F)
            
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
    if type(trial['hyperparameters']['learning rate']) == float:
        plt.figure()
        for idx in range(len(per_layer_rates[0])):
            rates = [trial['hyperparameters']['learning rate']*p[idx] for p in per_layer_rates]
            plt.plot(np.arange(len(rates))+1, rates, '.')
        plt.xlabel('Epoch')
        plt.ylabel('Magnitude of correction')
        plt.yscale('log')
        plt.savefig(os.path.join(filepath, 'rates.png'))
    elif type(trial['hyperparameters']['learning rate']) == list:
        plt.figure()
        for idx, lr in zip(range(len(per_layer_rates[0])), trial['hyperparameters']['learning rate']):
            rates = [lr*p[idx] for p in per_layer_rates]
            plt.plot(np.arange(len(rates))+1, rates, '.')
        plt.xlabel('Epoch')
        plt.ylabel('Magnitude of correction')
        plt.yscale('log')
        plt.savefig(os.path.join(filepath, 'rates.png'))
    plt.imsave(os.path.join(filepath, 'w.png'), Network.W.clone().squeeze().cpu().numpy())
    plt.imsave(os.path.join(filepath, 'wmask.png'), Network.W_mask.clone().squeeze().cpu().numpy())
    plt.imsave(os.path.join(filepath, 'mean_dW.png'), mean_dW.clone().squeeze().cpu().numpy())
    layer_rates_means = []
    for layer_idx, _ in enumerate(per_layer_rates[0]):
    	layer_rates_means.append(np.mean([l[layer_idx] for l in per_layer_rates]))
    spread_stdev = np.std([np.log10(lm) for lm in layer_rates_means])
    print('Spread stdev:', spread_stdev)
    
    
    
    
    
    
