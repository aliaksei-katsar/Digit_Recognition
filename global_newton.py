import numpy as np
from armijo import armijo
import functools


def global_newton(f: functools.partial, gradf: functools.partial, hessianf: functools.partial,
                  x, eps=1e-3, iter=1000, gamma=0.01, beta=0.5):
    alpha1 = 1e-8
    alpha2 = 1e-4
    p = 0.2

    current_grad = gradf(x)
    current_hess = hessianf(x)
    current_grad_norm = np.linalg.norm(current_grad)

    while current_grad_norm > eps and iter > 0:

        try:
            print(np.shape(current_hess))
            print(np.shape(current_grad))
            d = np.linalg.solve(current_hess, -1 * current_grad)
            if (-np.dot(current_grad, d) > np.min([alpha1, alpha2 * (np.power(current_grad_norm, p))]) *
                    current_grad_norm * np.linalg.norm(d)):
                s = d
                print("Newton used")
            else:
                s = -1 * current_grad
                print("Grad is used, system solved")
        except np.linalg.LinAlgError as e:
            s = -current_grad
            print("Grad is used")

        sigma = armijo(f, x, current_grad, s, gamma, beta)
        x = x + sigma * s
        current_grad = gradf(x)
        current_hess = hessianf(x)
        current_grad_norm = np.linalg.norm(current_grad)
        iter -= 1

        print("Current grad norm is: ", current_grad_norm)

    return x
