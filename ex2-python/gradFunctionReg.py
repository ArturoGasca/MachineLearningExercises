# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:18:14 2018

@author: in-qu
"""
import numpy as np
from sigmoid import sigmoid

def gradFunctionReg(theta, X, y, lambdaa):
    theta = theta.reshape(-1,1)
    y = y.reshape(-1,1)
    m = len(y)
    
    grad = 1 / m * (X.T @ (sigmoid(X @ theta) - y)) + lambdaa / m * theta
    grad[0] = 1 / m * (X[:, 0] @ (sigmoid(X @ theta) - y))
    grad = grad.flatten()
    
    return grad
    