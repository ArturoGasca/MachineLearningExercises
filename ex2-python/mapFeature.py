import numpy as np

def mapFeature(X1, X2, degree=6):    
    out = np.ones((len(X1),1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            values = X1 ** (i - j) * (X2 ** j)
            values = values.reshape(-1,1)
            out = np.append(out, values, 1)
    return out