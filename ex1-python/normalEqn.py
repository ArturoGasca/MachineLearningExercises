# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:08:32 2018

@author: in-qu
"""
from numpy.linalg import pinv
def normalEqn(X, y):
    theta = pinv(X.T @ X) @ X.T @ y
    return theta