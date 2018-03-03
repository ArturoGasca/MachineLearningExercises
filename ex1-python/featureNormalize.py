# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:15:04 2018

@author: in-qu
"""
import numpy as np
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    
    return [X_norm, mu, sigma]