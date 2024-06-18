import functools
import numpy as np
from armijo import armijo


def gradient_armijo(f: functools.partial, gradf: functools.partial, xk: np.array, eps=1e-6, iter = 1000) -> np.array:

    while np.linalg.norm(gradf(xk)) > eps and iter >= 0:
        gradf_x = gradf(xk)
        sigma = armijo(f, xk, gradf_x, -gradf_x, 0.5, 0.5)
        xk -= sigma * gradf_x
        iter -= 1
        print("Current norm is:")
        print(np.linalg.norm(gradf(xk)))

    return xk
