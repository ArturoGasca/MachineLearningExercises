# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:39:30 2018

@author: in-qu
"""
import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    
    return g