import function
import numpy as np
from armijo import armijo


def gradient_armijo(f: function, gradf: function, x: np.array, eps=1e-6) -> np.array:
    while gradf(x) > eps:
        gradf_x = gradf(x)
        sigma = armijo(f, x, gradf_x, -gradf_x, 0.01, 0.5)
        x -= sigma * gradf_x

    return x
