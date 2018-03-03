# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:50:31 2018

@author: in-qu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featureNormalize import featureNormalize
from normalEqn import normalEqn
from gradientDescent import gradientDescent

#Import dataset
dataset = pd.read_csv('ex1data2.txt', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
m = len(y)

# Normalize features
X, mu, sigma = featureNormalize(X)

# Add intercept column of 1s
X = np.insert(X, obj=0, values=1, axis=1)

# Gradient Descent params
num_iters = 400
theta = np.zeros((3,1))

# Perform the gradient descent algorithm with different values for alpha
alpha = 0.6
[theta1, J_history1] = gradientDescent(X,y,theta, alpha, num_iters)
alpha = 0.001
[theta2, J_history2] = gradientDescent(X,y,theta, alpha, num_iters)
alpha = 0.03
[theta3, J_history3] = gradientDescent(X,y,theta, alpha, num_iters)
alpha = 0.01
[theta4, J_history4] = gradientDescent(X,y,theta, alpha, num_iters)
alpha = 0.003
[theta5, J_history5] = gradientDescent(X,y,theta, alpha, num_iters)
alpha = 0.1
[theta, J_history] = gradientDescent(X,y,theta, alpha, num_iters)

# Visualize the performance of each gradient descent run
plt.plot(np.arange(0,len(J_history1)), J_history1, color='cyan')
plt.plot(np.arange(0,len(J_history2)), J_history2, color='magenta')
plt.plot(np.arange(0,len(J_history3)), J_history3, color='green')
plt.plot(np.arange(0,len(J_history4)), J_history4, color='blue')
plt.plot(np.arange(0,len(J_history5)), J_history5, color='red')
plt.plot(np.arange(0,len(J_history)), J_history, color='yellow')

# Predict new input
X = np.array([[1650, 3]])
X = (X - mu) / sigma
X = np.insert(X, obj=0, values=1, axis=1)
price = X @ theta1

#Import dataset again
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
m = len(y)

#Add intercept column of 1s
X = np.insert(X, obj=0, values=1, axis=1)

# Fit the linear function using the normal equation
theta = normalEqn(X, y)

#Predict new input
X = np.array([1, 1650, 3])
price = X @ theta
