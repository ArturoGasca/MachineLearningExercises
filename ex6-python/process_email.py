# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 23:04:03 2018

@author: in-qu
"""
from get_vocab_list import get_vocab_list
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from functools import reduce
stemmer = PorterStemmer()

def process_email(email_contents):
    vocablist = get_vocab_list()
    
    email_contents = email_contents.lower()
    
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    
    email_contents = re.sub('[0-9]+','number', email_contents)
    
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    
    email_contents = [processWord(x) for x in \
                      re.split("[" + re.escape(' @$/#.-:&*+=[]?!(){},\'">_<;%') + "]", email_contents) \
                      if x.strip()]

    word_indices = reduce(lambda x, y: appendWord(x, y, vocablist), \
                          email_contents, \
                          np.empty(0, dtype=int))
    
    return word_indices
    

def appendWord(word_indices, word, vocablist):
    if(not len(word) < 1):
        match = np.where(vocablist == word)
        word_indices = np.append(word_indices, match[0], 0)
    
    return word_indices

def processWord(word):
    word = re.sub('[^a-zA-Z0-9]', '', word)
    word = stemmer.stem(word)
    
    return word