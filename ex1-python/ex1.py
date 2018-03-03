# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:50:31 2018

@author: in-qu
"""
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotdata import plotData
from computecost import computeCost
from gradientDescent import gradientDescent
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#Import dataset
dataset = pd.read_csv('ex1data1.txt', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
m = len(y)

#Visualize the data
plotData(X, y)

# Add intercept column of 1s
X = np.insert(X, obj=0, values=1, axis=1)

#Initial values of theta
theta = np.zeros((2,1))

#Gradient Descent params
iterations = 4000 # 10000
alpha = 0.024 # 0.01

# Fit the linear function
[theta,J] = gradientDescent(X, y, theta, alpha, iterations)

# Visualize fitted linear function
plt.plot(X[:, 1], X@theta)

#Make predictions for new inputs
predict1 = [1, 3.5]@theta * 10000
predict2 = (np.array([1, 7])@theta) * 10000

#Visualize J values with different values for theta (3D surface plot)
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)

J_values = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        t = np.array([[t0], [t1]])
        J_values[i][j] = computeCost(X, t,y)

X, Y = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, J_values, cmap=cm.coolwarm,
                       linewidth=100, rcount=50, ccount=50)
plt.show()


#Visualize J values with different values for theta (2D contour plot)
plt.figure()
plt.contour(X,Y, J_values.T, np.logspace(-2,3,20))
plt.scatter(theta[0], theta[1], 20, color='red', marker='x')
plt.show()