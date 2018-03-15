# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 22:18:00 2018

@author: in-qu
"""
import numpy as np

def email_features(word_indices):
    n = 1899
    x = np.zeros(n)
    x[word_indices] = 1
    
    return x