# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:10:57 2018

@author: in-qu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:06:32 2018

@author: in-qu
"""

import matplotlib.pyplot as plt
import numpy as np

from mapFeature import mapFeature

def plotDecisionBoundary(theta, X, y):
    
    if(np.size(X, 1) <= 3):
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 2]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y)
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))
        
        for i,ui in enumerate(u):
            for j,vi in enumerate(v):
                z[i, j] = mapFeature(ui.reshape(-1,1), vi.reshape(-1,1)) @ theta
                
        plt.contour(u,v, z.T, 0)
        return z