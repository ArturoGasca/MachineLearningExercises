import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from mapFeature import mapFeature

dataset = pd.read_csv('ex2data2.txt', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

plotData(X, y)

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6, include_bias=False)
X = poly_reg.fit_transform(X)

from sklearn.linear_model import LogisticRegression
lambdaa = 10
classifier = LogisticRegression(random_state=0, tol=1e-6, C=1/lambdaa)
classifier.fit(X, y)
p = classifier.predict(X)

sk_theta = np.array([classifier.intercept_])
sk_theta = np.append(sk_theta, classifier.coef_.flatten())

X = np.insert(X, 0, 1, axis=1)

z = plotDecisionBoundary(sk_theta, X, y)
p = classifier.predict(X)

from sklearn.metrics import accuracy_score
sk_accuracy = accuracy_score(y, p) * 100