# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('ex1data1.txt',  header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Linear Regression (it handles the feature scaling by itself)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

sk_theta = np.array([[regressor.intercept_], regressor.coef_])

# Predict new inputs
sk_prediction1 = regressor.predict(3.5) * 10000
sk_prediction2 = regressor.predict(7) * 10000

#Visualize the fitted linear function
plt.plot(X, regressor.predict(X), color="green")