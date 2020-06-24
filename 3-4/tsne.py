import h5py
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from random import random

h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')


global_features = np.array(h5f_data['dataset_1'])
global_labels = np.array(h5f_label['dataset_1'])

h5f_data.close()
h5f_label.close()

class_names = ["bulbasaur", "charmander", "mewtwo", "pikachu", "squirtle"]

ss = StandardScaler()
scaled_features = ss.fit_transform(global_features)

tsne = TSNE(n_components=2, perplexity=10)
tsne_data = tsne.fit_transform(scaled_features)
tx = tsne_data[:, 0]
ty = tsne_data[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


plt.figure(figsize=(8, 7))

for index, label in enumerate(class_names):
    color = (random(), random(), random(), 1)
    plt.scatter(tx[global_labels == index], ty[global_labels == index], color=color, label=label)

plt.legend()
plt.show()
