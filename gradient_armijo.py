import functools
import numpy as np
from armijo import armijo
from power_wilfe import power_wilfe


def gradient_armijo(f: functools.partial, gradf: functools.partial, xk: np.array, eps=1e-6, iter = 1000) -> np.array:

    while np.linalg.norm(gradf(xk)) > eps and iter >= 0:
        gradf_x = gradf(xk)
        #sigma = power_wilfe(f, gradf, xk, gradf_x, -gradf_x, 0.0001)
        #sigma = armijo(f, xk, gradf_x, -gradf_x, 0.01, 0.5)
        sigma = 0.2
        '''if(np.linalg.norm(gradf(xk)) > 1):
            sigma = 0.2#armijo(f, xk, gradf_x, -gradf_x, 0.01, 0.5)
        else:
            sigma = 0.5'''
        xk -= sigma * gradf_x
        iter -= 1
        print("iteration: ", 1e6 - iter)
        print("Current norm is:")
        print(np.linalg.norm(gradf(xk)))
        print("Current function is:")
        print(np.linalg.norm(f(xk)))

    return xk
