
Directories overview --

The trials directory contains scripts used to set up a network and run trials. The script run_trials.py can run a list of trials to reproduce the datasets shown in all figures exept figure 4 in the paper. The script run_trials_sweep.py can sweep the parameter p of a network to reproduce the dataset used for figure 4.

The results directory contains the output of the scripts in the trials folder.

The framework directory contains the code used to implement the equilibrium propagation framework. eqp.py contains the code implementing a network and the algorithm to train it. datasets.py contains a list of classes used to download and parse datasets that were trained on in the paper.

The datasets directory stores datasets that were trained on during the paper. These datasets should be automatically downloaded the first time they are used, if they are not already present in this directory.


How to reproduce results in paper --

Datasets used in the paper and the scripts used to generate them can be found at github.com/jgammell/eqp_paper_data.

The file run_trials.py can reproduce all datasets except the one used for figure 3. The variable trials on line 304 contains a list of trial configuration settings to run. Above this is a list of configuration settings for trials used to generate datasets used in the paper; to reproduce a list of these datasets one can replace the variable by the list of settings corresponding to these datasets. Note that the folder corresponding to 'folder' must exist in the results directory.
Figures 2 and 3 used mnist_3layer_mlff for the MLFF network with one global learning rate, mnist_3layer_mlffpl for the MLFF network with per-layer rates, and mnist_3layer_swni for the SW network.
Figure 5 used mnist_5layer_mlff for the MLFF network and mnist_5layer_swni for the SW network.
Table 1 used, in order of listing in the table, diabetes_5layer_mlff, diabetes_5layer_swni, diabetes_8layer_mlff, diabetes_8layer_swni, wine_5layer_mlff, wine_5layer_swni, wine_8layer_mlff, wine_8layer_swni, mnist_3layer_mlff, mnist_3layer_swni, mnist_5layer_mlff, mnist_5layer_swni, fmnist_5layer_mlff, fmnist_5layer_swni.

The file run_trials_sweep.py can reproduce the dataset used to generate figure 4. The variable trial on line 37 contains the initial settings for the network whose p parameter will be swept. The setting 'bypass p' will be ignored and for each network it will instead be initialized to the values resulting from line 39. Note that the folder 'mnist_5layer_sweep' must exist in the results directory and be empty; the folder corresponding to 'folder' in the trial settings is ignored.
Figure 4 used mnist_5layer_swni.

The output of a trial is structured as follows:
init.pickle is a dictionary where 'trial settings' is the settings dictionary for the trial, 'initial weight' is the initial weight matrix of the network, and 'initial mask' is the mask for this weight matrix where elements (i, j) is 1 if neurons i and j are connected, and else 0.
log.txt contains anything that is printed to the terminal during a trial, including success/failure indications, time taken to do execute various tasks, error rates after each epoch, network settings, and log-spread after training.
e%d.pickle are dictionaries saved after each epoch where 'training error' gives the training error, 'test error' gives the test error, 'per-layer rates' gives a list of the per-layer rates after each batch, 'mean dW' gives the mean weight matrix over all epochs leading up to and including that epoch, and 'states' contains a tuple with element 0 containing the number of states saturated at 0 after the free phase and element 1 containing the number of states saturated at 1.
final_network.pickle contains the network class after training is completed; this can be useful if one wants to carry on training from where it left off, or to validate any aspects of the network after training.
After running run_trial.py, these datapoints will be stored in the folder corresponding to 'folder' in the trial settings. After running run_trial_sweep.py, these datapoints will be stored in the folder mnist_5layer_sweep/trial_p%0.6f where %0.6f is the bypass probability p.


How to use --

Configuring network:
 1) Create a topology settings dictionary:
topology = \
{
  'layer sizes': , # List containing number of neurons in each layer, e.g. for network with 8 input neurons, 2 7-neuron hidden layers, 6 output neurons, input should be [8, 7, 7, 6]
  'network type': , # Possible values: 'MLFF' for multilayer feedforward topology, 'SW' for SW topology with fully-connected hidden layers, 'SW_no_intra' for SW topology described in paper
  'bypass p': , # Probability with which to randomly rewire each pre-existing connection in the network; ignored with MLFF topology
  'bypass mag': # New layer-skipping connections will have weights initially drawn from uniform distribution U[-a, a] where a is this argument
}
 2) Create a hyperparameters settings dictionary:
hyperparameters = \
{
  'learning rate': , # Learning rate of network. Can be a scalar or a list if per-layer rates are desired; in the latter case the list must have l-1 elements where l is the length of topology['layer sizes'].
  'epsilon': , # Step size for the forward Euler approximation of the network dynamics.
  'beta': , # Extent to which output layer is perturbed towards target output during weakly-clamped phase.
  'free iterations': , # Number of iterations of forward Euler approximation during free phase.
  'weakly clamped iterations': # Number of iterations of forward Euler approximation during weakly-clamped phase.
}
 3) Create a general configuration settings dictionary:
configuration = \
{
  'batch size': , # Number of examples per training batch.
  'device': , # Pytorch device on which to train network. Tested using 'cpu' for CPU and 'cuda' for NVIDIA GPU.
  'seed': # Initial seed for Pytorch and Numpy random number generators. Trials with identical configurations should be identical when they have the same seed.
}
4) Choose a dataset. Implemented options are datasets.MNIST for MNIST, datasets.FashionMNIST for Fashion MNIST, datasets.Diabetes for diabetes sklearn toy dataset, and datasets.Wine for sklearn wine toy dataset.

Training network:
1) Initialize network by creating instance of eqp.Network class: network=eqp.Network(topology, hyperparameters, configuration, dataset).
2) Train network by calling network.train_epoch(). After calling this function, the variable network.training_error will store the training error computed throughout that epoch (i.e. error on each batch prior to running weakly-clamped phase).
3) Compute training error all at once by calling network.calculate_training_error(), after which the variable network.true_training_error will contain the training error. This value is more-comparable to the test error than the value computed in step 2 because it is not taken throughout an epoch, but all at once at the end.
4) Compute test error by calling network.calculate_test_error(), after which the variable network.test_error will contain the test error.
5) The variable network.per_layer_rates contains a list of the per-layer-rates computed during each batch of the most-recent epoch of training, as seen in figure 3 of the paper. These are not scaled by the learning rate.
6) The variable network.mean_dW contains the mean magnitude correction matrix over the last epoch of training, as seen in figure 5 of the paper.
