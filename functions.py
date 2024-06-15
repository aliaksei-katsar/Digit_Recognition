import numpy as np


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def loss_function(x: np.array, y: np.array, wb: np.array) -> float:
    w = wb[:-1]
    b = wb[-1]
    return -1 / len(x) * np.sum(y[i] * np.log(sigmoid(np.dot(x[i], w) + b))
                                + (1 - y[i]) * np.log(1 - sigmoid(np.dot(x[i], w) + b))
                                for i in range(len(x)))


def grad_loss_function(x: np.array, y: np.array, wb: np.array) -> np.array:
    w = wb[:-1]
    b = wb[-1]
    return -1 / len(x) * np.sum((y[i] - 1 / (sigmoid(np.dot(w, x) + b))) * np.array([x[i], 1])
                                for i in range(len(x)))
