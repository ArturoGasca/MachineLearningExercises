
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plotData import plotData
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary

dataset = pd.read_csv('ex2data1.txt', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

plotData(X, y)

classifier = LogisticRegression(random_state=0, tol=1e-6, C=10000)
classifier.fit(X, y)
p = classifier.predict(X)

sk_theta = np.array([classifier.intercept_])
sk_theta = np.append(sk_theta, classifier.coef_.flatten())

X = np.insert(X, 0, 1, axis=1)
plotDecisionBoundary(sk_theta, X, y)

from sklearn.metrics import accuracy_score
sk_accuracy = accuracy_score(y, p) * 100
