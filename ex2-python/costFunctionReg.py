# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:08:57 2018

@author: in-qu
"""
import numpy as np
from sigmoid import sigmoid
def costFunctionReg(theta, X, y, lambdaa):
    theta = theta.reshape(-1,1)
    y = y.reshape(-1,1)
    m = len(y)
    J = -1 / m * (y.T @ np.log(sigmoid(X @ theta)) + (1 - y.T) @ np.log(1 - sigmoid(X @ theta))) + \
        lambdaa / (2*m) * (theta[1:].T @ theta[1:])
    
    return J
    