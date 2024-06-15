import numpy as np


def armijo(f: function, x: np.array, gradf: np.array, s: np.array, gamma=0.01, beta=0.5) -> float:
    sigma = 1
    fx = f(x)
    slope = gamma * np.dot(gradf, s)
    while (f(x + sigma + s) - fx <= sigma * slope):
        sigma *= beta
    return sigma
