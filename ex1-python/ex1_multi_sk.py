# Import libraries
import numpy as np
import pandas as pd

#Import dataset
dataset = pd.read_csv('ex1data2.txt',  header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

# Linear Regression (it handles the feature scaling by itself)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

sk_theta = np.array([regressor.intercept_])
sk_theta = np.concatenate((sk_theta, regressor.coef_))

# Predict new input
X = np.array([[1650, 3]])
sk_price = regressor.predict(X)