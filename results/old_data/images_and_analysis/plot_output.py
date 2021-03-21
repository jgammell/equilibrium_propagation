#%%
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pickle
import os
import numpy as np
import matplotlib
import torch
os.chdir(os.path.join(os.getcwd(), '..'))
import sys
sys.path.append(os.getcwd())
from framework import eqp
from framework import datasets
os.chdir(os.path.join(os.getcwd(), 'results'))

print('Loading MLFF data.')
with open(os.path.join(os.getcwd(), 'diabetes_5layer_MLFF', 'init.pickle'), 'rb') as F:
    mlff_initial_settings = pickle.load(F)
print(mlff_initial_settings)
matplotlib.image.imsave('mlff_initial_weight.png', np.abs(mlff_initial_settings['initial weight']))
matplotlib.image.imsave('mlff_initial_mask.png', np.abs(mlff_initial_settings['initial mask']))
with open(os.path.join(os.getcwd(), 'diabetes_5layer_MLFF', 'final_network.pickle'), 'rb') as F:
    Network = pickle.load(F)
    dW_avg = Network.mean_dW.clone().cpu().squeeze().numpy()
    final_W = Network.W.clone().cpu().squeeze().numpy()
matplotlib.image.imsave('mlff_mean_dw_bin.png', np.where(dW_avg != 0, 1, 0), vmin=0, vmax=1)
matplotlib.image.imsave('mlff_mean_dw.png', dW_avg, vmin=0, vmax=.1*np.max(dW_avg))
matplotlib.image.imsave('mlff_final_w.png', np.abs(final_W), vmin=0, vmax=np.max(np.abs(final_W)))
mlff_training_error = []
mlff_test_error = []
mlff_perlayer = []
for i in np.arange(100):
    with open(os.path.join(os.getcwd(), 'diabetes_5layer_MLFF', 'e%d.pickle'%(i)), 'rb') as F:
        Data = pickle.load(F)
    mlff_training_error.append(Data['training error'])
    mlff_test_error.append(Data['test error'])
    mlff_perlayer.append(Data['per-layer rates'])
print('Loading SW data.')
with open(os.path.join(os.getcwd(), 'diabetes_5layer_SW', 'init.pickle'), 'rb') as F:
    sw_initial_settings = pickle.load(F)
matplotlib.image.imsave('sw_initial_weight.png', np.abs(sw_initial_settings['initial weight']))
matplotlib.image.imsave('sw_initial_mask.png', np.abs(sw_initial_settings['initial mask']))
with open(os.path.join(os.getcwd(), 'diabetes_5layer_SW', 'final_network.pickle'), 'rb') as F:
    Network = pickle.load(F)
    dW_avg = Network.mean_dW.clone().cpu().squeeze().numpy()
    final_W = Network.W.clone().cpu().squeeze().numpy()
matplotlib.image.imsave('sw_mean_dw_bin.png', np.where(dW_avg != 0, 1, 0), vmin=0, vmax=1)
matplotlib.image.imsave('sw_mean_dw.png', dW_avg, vmin=0, vmax=.1*np.max(dW_avg))
matplotlib.image.imsave('sw_final_w.png', np.abs(final_W), vmin=0, vmax=np.max(np.abs(final_W)))
print(sw_initial_settings)
sw_training_error = []
sw_test_error = []
sw_perlayer = []
for i in np.arange(100):
    with open(os.path.join(os.getcwd(), 'diabetes_5layer_SW', 'e%d.pickle'%(i)), 'rb') as F:
        Data = pickle.load(F)
    sw_training_error.append(Data['training error'])
    sw_test_error.append(Data['test error'])
    sw_perlayer.append(Data['per-layer rates'])

(fig, ax) = plt.subplots(1, 1)
ax.plot(range(len(mlff_training_error)), mlff_training_error, '-', color='blue', label='MLFF')
ax.plot(range(len(mlff_test_error)), mlff_test_error, '--', color='blue')
ax.plot(range(len(sw_training_error)), sw_training_error, '-', color='red', label='SW')
ax.plot(range(len(sw_test_error)), sw_test_error, '--', color='red')
fig.savefig('error_rates.png')

(fig, ax) = plt.subplots(1, 1)
ax.plot(np.arange(len(mlff_training_error)), mlff_training_error, color='blue', label='MLFF')
ax.plot(np.arange(len(mlff_test_error)), mlff_test_error, '--', color='blue')
ax.plot(np.arange(len(sw_training_error)), sw_training_error, color='red', label='SW')
ax.plot(np.arange(len(sw_test_error)), sw_test_error, '--', color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error rate')
ax.set_title('Error rate vs. epoch')
ax.legend()

colors = ['red', 'blue', 'green', 'orange', 'purple']
(fig, ax) = plt.subplots(1, 2, sharey=True)
for i, color in zip(range(len(mlff_perlayer[0])), colors):
    rates = [.02*p[i] for p in mlff_perlayer]
    ax[0].plot(range(len(rates)), rates, '.', color=color)
for i, color in zip(range(len(sw_perlayer[0])), colors):
    rates = [.02*p[i] for p in sw_perlayer]
    ax[1].plot(range(len(rates)), rates, '.', color=color)
ax[0].set_xlabel('Epoch')
ax[1].set_xlabel('Epoch')
ax[0].set_ylabel('RMS correction magnitude')
ax[0].set_title('MLFF')
ax[1].set_title('SW')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.savefig('layer_rates.png')
