
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from plotData import plotData
from costFunction import costFunction
from gradFunction import gradFunction
from sigmoid import sigmoid
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

dataset = pd.read_csv('ex2data1.txt', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

plotData(X, y)

m, n = X.shape

X = np.insert(X, 0, 1, axis=1)
initial_theta = np.zeros((n+1, 1))

test_theta = np.array([-24, 0.2, 0.2])

theta, cost, *res = opt.fmin_bfgs(costFunction, \
                                  test_theta, \
                                  gradFunction, \
                                  (X, y), \
                                  maxiter=400, \
                                  full_output=True)

theta, *res= opt.fmin_tnc(costFunction, \
                          test_theta, \
                          gradFunction, \
                          (X, y))

plotDecisionBoundary(theta, X, y)

prob = sigmoid(np.array([1, 45, 85] @ theta))

p = predict(theta, X)

#from sklearn.metrics import accuracy_score
accuracy = np.mean(p == y) * 100 #accuracy = accuracy_score(y, p)
