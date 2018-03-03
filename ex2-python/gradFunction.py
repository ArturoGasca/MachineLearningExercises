# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:36:18 2018

@author: in-qu
"""
from sigmoid import sigmoid

def gradFunction(theta, X, y):
    theta = theta.reshape(-1,1)
    y = y.reshape(-1,1)
    m = len(y)
    grad = 1 / m * X.T @ (sigmoid(X @ theta) - y)
    grad = grad.flatten()
    
    return grad
