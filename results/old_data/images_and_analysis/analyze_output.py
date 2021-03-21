import pickle
import numpy as np
import os
from matplotlib import pyplot as plt

folders = ['mnist_3layer_swni']

for folder in folders:
    training_errors = []
    test_errors = []
    layer_rates = []
    #with open(os.path.join(os.getcwd(), folder, 'init.pickle'), 'rb') as F:
    #    initial_settings = pickle.load(F)
    #learning_rate = initial_settings['hyperparameters']['learning rate']
    for epoch in np.arange(107):
        with open(os.path.join(os.getcwd(), folder, 'e%d.pickle'%(epoch)), 'rb') as F:
            Data = pickle.load(F)
        training_errors.append(Data['training error'])
        test_errors.append(Data['test error'])
        #layer_rates.append([learning_rate*lr for lr in Data['per-layer rates']])
    #layer_rates_means = []
    #for layer_idx in np.arange(len(layer_rates[0])):
    #    layer_rates_means.append(np.mean([l[layer_idx] for l in layer_rates]))
    #spread_stdev = np.std([np.log10(lm) for lm in layer_rates_means])
    #spread_range = np.max([-np.log10(lm) for lm in layer_rates_means]) - np.min([-np.log10(lm) for lm in layer_rates_means])
    #print('Folder: %s'%(folder))
    #print('\tMean layer rates:', layer_rates_means)
    #print('\tSpread stdev:', spread_stdev)
    #print('\tSpread range:', spread_range)
    #print('\tFinal training error:', training_errors[-1])
    #print('\tFinal test error:', test_errors[-1])
    plt.figure()
    plt.plot(np.arange(107), training_errors)
    plt.plot(np.arange(107), test_errors, '--')
    plt.savefig(os.path.join(os.getcwd(), folder, 'error.png'))
