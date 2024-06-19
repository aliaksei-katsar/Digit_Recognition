
import function
import numpy as np


def armijo(f: function, x: np.array, gradf: np.array, s: np.array, gamma=0.01, beta=0.5) -> float:
    sigma = 1
    fx = f(x)
    slope = gamma * np.dot(gradf.flatten(), s.flatten())
    while f(x + sigma * s) - fx >= sigma * slope and sigma > gamma:
        #print(f(x + sigma * s) - fx)
        #print(sigma * slope)
        sigma *= beta
    #print(f(x + sigma * s) - fx)
    #print(sigma * slope)
    if(sigma < gamma):
        print("No Armijo")
    return sigma
