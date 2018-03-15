import matplotlib.pyplot as plt

def plot_data(X, y):
    pos = X[y == 1]
    neg = X[y == 0]
    plt.scatter(pos[:,0], pos[:,1], marker='+', c='black')
    plt.scatter(neg[:,0], neg[:,1], marker='o', c='yellow', linewidths=1, edgecolors='black')