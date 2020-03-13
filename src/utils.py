import numpy as np
from tqdm import tqdm
import iisignature

def _transform(X):
    if len(X) == 1:
        return np.array([[-X[0, 0], X[0, 1]], [X[0, 0], X[0, 1]]])
    new_X = [[-X[1, 0], X[0, 1]]]
    for x_past, x_future in zip(X[:-1], X[1:]):
        new_X.append(x_past)
        new_X.append([x_past[0], x_future[1]])
        
    new_X.append(X[-1])
    
    return np.array(new_X)


def transform(X):
    timeline = X[:, 0]
    
    new_X = [None]
    
    for x in X.T[1:]:
        projection = np.c_[timeline, x]
        new_projection = _transform(projection)
        new_X[0] = new_projection[:, 0]
        new_X.append(new_projection[:, 1])

    return np.array(new_X).T
        