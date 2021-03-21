import pickle
import os
from matplotlib import pyplot as plt
import numpy as np

swni_folders = ['swni_s%d'%(i) for i in range(5)]
mlff_folders = ['mlff_s%d'%(i) for i in range(2)]

plt.figure()
plt.xlabel('epoch')
plt.ylabel('error')

for folder in swni_folders:
  training_errors = []
  test_errors = []
  for epoch in np.arange(10):
    with open(os.path.join(os.getcwd(), folder, 'e%d.pickle'%(epoch)), 'rb') as F:
      Data = pickle.load(F)
    training_errors.append(Data['training error'])
    test_errors.append(Data['test error'])
  plt.plot(range(1, 11), training_errors, color='blue')
  plt.plot(range(1, 11), test_errors, '--', color='blue')
for folder in mlff_folders:
  training_errors = []
  test_errors = []
  for epoch in np.arange(10):
    with open(os.path.join(os.getcwd(), folder, 'e%d.pickle'%(epoch)), 'rb') as F:
      Data = pickle.load(F)
    training_errors.append(Data['training error'])
    test_errors.append(Data['test error'])
  plt.plot(range(1, 11), training_errors, color='red')
  plt.plot(range(1, 11), test_errors, '--', color='red')
plt.savefig('error.png')
