# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:50:08 2018

@author: in-qu
"""
from plot_data import plot_data
import numpy as np
import matplotlib.pyplot as plt
def visualize_boundary(X, y, model):
    plot_data(X, y)    
    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).T
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)    
    
    for i in range(np.size(X1, 1)):
        this_X = np.array([X1[:, i], X2[:, i]])
        vals[:, i] = model.predict(this_X.T)
    plt.contour(X1, X2, vals, 0.5)