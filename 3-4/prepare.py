from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import cv2
import os
import h5py
from features import fd_hu_moments, fd_haralick, fd_histogram, fd_4, fd_5, fd_6


images_per_class = 250
fixed_size = tuple((500, 500))
train_path = os.path.join('data', 'train')
output_dir = 'output'
h5_data = os.path.join(output_dir, 'data.h5')
h5_labels = os.path.join(output_dir, 'labels.h5')

train_labels = os.listdir(train_path)

train_labels.sort()
print(train_labels)

global_features = []
labels = []

for training_name in train_labels:
    current_dir = os.path.join(train_path, training_name)
    images = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f)) and f.endswith(".jpg")]

    current_label = training_name

    for file in images:
        file_path = os.path.join(current_dir, file)

        image = cv2.imread(file_path)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments=fd_hu_moments(image)
        fv_histogram=fd_histogram(image)
        fv_haralick = fd_haralick(image)
        fv_4 = fd_4(image)
        fv_5= fd_5(image)
        fv_6 = fd_6(image)
        global_feature = np.hstack([fv_6, fv_5])

        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

print("[STATUS] training Labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

rescaled_features = global_features
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")

