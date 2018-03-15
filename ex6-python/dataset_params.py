# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:23:20 2018

@author: in-qu
"""
import numpy as np
from sklearn.svm import SVC

def dataset_params(X, y, Xval, yval):
    
    #gamma_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gamma_val = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    c_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    error_val = np.zeros((len(gamma_val), len(c_val)))
    
    for i in range(len(gamma_val)):
        for j in range(len(c_val)):
            current_gamma = gamma_val[i]
            current_c = c_val[j]
            model = SVC(C=current_c, gamma=current_gamma, random_state=0)
            model.fit(X, y)
            predictions = model.predict(Xval)
            error_val[i, j] = np.mean(predictions != yval)
            
    gamma_index, c_index = np.argwhere(error_val==np.min(error_val))[0]
    
    gamma = gamma_val[gamma_index]
    C = c_val[c_index]
    min_error = np.min(error_val)
    
    return [gamma, C, min_error]