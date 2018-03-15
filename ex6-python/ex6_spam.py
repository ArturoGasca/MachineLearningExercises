# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:35:00 2018

@author: in-qu
"""
import numpy as np

from get_vocab_list import get_vocab_list
from process_email import process_email
from email_features import email_features
from scipy.io import loadmat
from sklearn.svm import SVC

with open('emailSample1.txt', 'r') as myfile:
    data = myfile.read()
word_indices = process_email(data)
features = email_features(word_indices)

dataset = loadmat('spamTrain.mat')
#dataset = loadmat('ex6data1.mat')
X = dataset['X']
y = dataset['y'].astype(int).reshape(-1)

C = 0.1
model = SVC(C, kernel='linear')
model.fit(X, y)
p = model.predict(X)

train_accuracy = np.mean(p == y)

dataset = loadmat('spamTest.mat')
Xtest = dataset['Xtest']
ytest = dataset['ytest'].astype(int).reshape(-1)
p = model.predict(Xtest)

test_accuracy = np.mean(p == ytest)

theta = np.array(model.coef_.flatten())
sorted_indices = np.argsort(theta)[::-1]

vocablist = get_vocab_list()

for idx, i in enumerate(sorted_indices[:15]):
    print('{} \t {}'.format(vocablist[i], theta[i]))
    

filename = 'emailSample4.txt'
with open(filename, 'r') as myfile:
    data = myfile.read()
word_indices = process_email(data)
x = email_features(word_indices)
x = x.reshape(1,-1)
p = model.predict(x)

print('Processed {}\n\nSpam Classification: {}\n'.format(filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')