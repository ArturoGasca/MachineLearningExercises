import numpy as np
from computecost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    y = y.reshape(-1,1)
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(0, num_iters):
        theta = theta - alpha * (1/m) * (X.T @ (X @ theta - y))
        
        J_history[i] = computeCost(X, theta, y)
        
    return [theta, J_history]