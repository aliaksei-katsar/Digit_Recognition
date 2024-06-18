
from functools import partial
import numpy as np
import functions as f
import gradient_armijo as gar
import data_loading
import global_newton as gn


x_train, y_train, x_test, y_test = data_loading.load_data()

x_start = np.zeros(len(x_train[0]) + 1)
loss = partial(f.loss_function, x = x_train, y = y_train)
grad_loss = partial(f.grad_loss_function, x = x_train, y = y_train)
hessian_loss = partial(f.hessian_loss_function, x = x_train)
sol = gar.gradient_armijo(loss, grad_loss, x_start, 1e-3)
#sol = gn.global_newton(loss, grad_loss, hessian_loss, x_start)

c = 0
for (xi, yi) in zip(x_test, y_test):
    v = np.dot(xi, sol[:-1]) + sol[-1]
    if(v >= 0):
        if(yi == 1):
            c += 1
    else:
        if(yi == 0):
            c += 1
print("right decisions: ", c)
print("all decisions: ", len(x_test))
