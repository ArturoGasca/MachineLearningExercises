import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from plotData import plotData
from costFunctionReg import costFunctionReg
from gradFunctionReg import gradFunctionReg
from sigmoid import sigmoid
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict
from mapFeature import mapFeature

# Load dataset
dataset = pd.read_csv('ex2data2.txt', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

# Visualize the data
plotData(X, y)

# Polynomial features
X = mapFeature(X[:, 0], X[:, 1])

lambdaa = 1

initial_theta = np.zeros((np.size(X, 1), 1))
lambdaa = 100

# Fit the decision boundary using two optimization functions
theta, J, *res = opt.fmin_bfgs(costFunctionReg, \
                               initial_theta, \
                               gradFunctionReg, \
                               args=(X, y, lambdaa), \
                               maxiter=400, \
                               #200 for exact results
                               full_output=True)

theta, *res= opt.fmin_tnc(costFunctionReg, \
                          initial_theta, \
                          gradFunctionReg, \
                          args=(X, y, lambdaa))
J = costFunctionReg(theta, X, y, lambdaa)

# Visualize the decision boundary
z = plotDecisionBoundary(theta, X, y)
p = predict(theta, X)

# Evaluate the model
from sklearn.metrics import accuracy_score
#accuracy = np.mean(p == y) * 100 
accuracy = accuracy_score(y, p) * 100