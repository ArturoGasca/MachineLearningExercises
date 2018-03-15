import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat
from plot_data import plot_data
from visualize_boundary_linear import visualize_boundary_linear
from visualize_boundary import visualize_boundary
from dataset_params import dataset_params

dataset = loadmat('ex6data1.mat')
X = dataset['X']
y = dataset['y'].astype(int).reshape(-1)

#plot_data(X, y)

from sklearn.svm import SVC
C = 1
classifier = SVC(C, 'linear', tol=1e-3, random_state=0)
classifier.fit(X, y)
#visualize_boundary_linear(X, y, classifier)


dataset = loadmat('ex6data2.mat')
X = dataset['X']
y = dataset['y'].astype(int).reshape(-1)

l = 1
C = 1 / l
gamma=100
classifier = SVC(C, 'rbf', tol=1e-3, random_state=0, gamma=gamma)
classifier.fit(X, y)
#visualize_boundary(X, y, classifier)



dataset = loadmat('ex6data3.mat')
X = dataset['X']
y = dataset['y'].astype(int).reshape(-1)

Xval = dataset['Xval']
yval = dataset['yval'].astype(int).reshape(-1)


C = 1
gamma, C, err = dataset_params(X, y, Xval, yval)
classifier = SVC(C, 'rbf', tol=1e-3, random_state=0, gamma=gamma)
classifier.fit(X, y)
visualize_boundary(X, y, classifier)
