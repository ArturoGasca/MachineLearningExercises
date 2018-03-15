# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:38:03 2018

@author: in-qu
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_data import plot_data

def visualize_boundary_linear(X, y, model):
    w = model.coef_.reshape(-1)
    b = model.intercept_
    xp = np.linspace(np.min(X[:,0]), np.max(X[:, 0]), 100)
    yp = - (w[0]*xp + b) / w[1]
    print(yp)    
    plot_data(X,y)
    plt.plot(xp, yp)