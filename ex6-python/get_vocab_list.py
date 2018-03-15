# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:34:21 2018

@author: in-qu
"""
import numpy as np
import re

def get_vocab_list():
    fname = 'vocab.txt'
    with open(fname) as f:
        content = f.readlines()
        content = [a.split("\t")[1] for a in content]
        content = [re.sub(r'\W+', '', a) for a in content]
        content = np.array(content)
        
        return content