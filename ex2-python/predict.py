# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:20:49 2018

@author: in-qu
"""
from sigmoid import sigmoid
import numpy as np
def predict(theta, X):
    p = np.round(sigmoid(X @ theta))
    
    return p