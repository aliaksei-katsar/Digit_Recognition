from tensorflow.keras.datasets import mnist
import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.array([arr.flatten() for arr in x_train])
    y_train = np.array(y_train)

    mask = np.isin(y_train, [0, 1])
    filtered_x_train = x_train[mask]
    filtered_y_train = y_train[mask]

    x_test = np.array([arr.flatten() for arr in x_test])
    y_test = np.array(y_test)

    mask = np.isin(y_test, [0, 1])
    filtered_x_test = x_test[mask]
    filtered_y_test = y_test[mask]

    filtered_x_train = filtered_x_train.astype(np.float32) / 255.0
    filtered_x_test = filtered_x_test.astype(np.float32) / 255.0

    return filtered_x_train, filtered_y_train, filtered_x_test, filtered_y_test
