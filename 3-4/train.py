import h5py
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from features import  fd_haralick, fd_4

fixed_size = tuple((500, 500))

warnings.filterwarnings('ignore')


num_trees = 100
test_size = 0.10
seed = 9
train_path = "data/train"
test_path = "data/test"
result_path = "data/result"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
scoring = "accuracy"

train_labels = os.listdir(train_path)

train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

models = [('LR', LogisticRegression(random_state=seed)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=seed)),
          ('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)),
          ('NB', GaussianNB())]

results = []
names = []

h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

import matplotlib.pyplot as plt

clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf.fit(trainDataGlobal, trainLabelsGlobal)

for idx, file in enumerate(glob.glob(test_path + "/*.jpg")):
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)

    fv_histogram = fd_haralick(image)
    fv_4=fd_4(image)

    global_feature = np.hstack([fv_histogram, fv_4])

    rescaled_feature = global_feature

    prediction = clf.predict(rescaled_feature.reshape(1, -1))[0]

    cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    plt.imsave(os.path.join(result_path, '%s.png' % idx), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))