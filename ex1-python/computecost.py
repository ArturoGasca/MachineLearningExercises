# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 18:15:48 2018

@author: in-qu
"""

def computeCost(X, theta, y):
    y = y.reshape(-1,1)
    m = len(y)
    J = 1 / (2 * m) * ((X @ theta - y).T @ (X @ theta - y))
    
    return J