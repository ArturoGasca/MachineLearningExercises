# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:36:18 2018

@author: in-qu
"""
import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    theta = theta.reshape(-1,1)
    y = y.reshape(-1,1)
    m = len(y)
    J = -1 / m * (y.T @ np.log(sigmoid(X @ theta)) + (1 - y.T) @ np.log(1 - sigmoid(X @ theta)))
    
    return J
    
    