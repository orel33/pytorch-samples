# -*- coding: utf-8 -*-
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import numpy as np
import math


# Create random input and output data (array of 2000 elements)
x = np.linspace(-math.pi, math.pi, 2000)
print("x=", x)
y = np.sin(x)
print("y=", y)

# Randomly initialize weights (scalar values)
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
print("a=", a, "b=", b, "c=", c, "d=", d)

# compute predicted y (array of 2000 elements)
# y = a + b x + c x^2 + d x^3
y_pred = a + b * x + c * x ** 2 + d * x ** 3
print("y_pred=", y_pred)

# compute loss (scalar value)
# loss = sum of squared differences (squared distance between y and y_pred)
loss = np.square(y_pred - y).sum() 
print("loss=", loss)

# compute gradients of a, b, c, d with respect to loss
grad_y_pred = 2.0 * (y_pred - y)        # explain this
grad_a = grad_y_pred.sum()              # d y_pred / d a
grad_b = (grad_y_pred * x).sum()        # d y_pred / d b
grad_c = (grad_y_pred * x ** 2).sum()   # d y_pred / d c
grad_d = (grad_y_pred * x ** 3).sum()   # d y_pred / d d

# update weights
learning_rate = 1e-6
a -= learning_rate * grad_a
b -= learning_rate * grad_b
c -= learning_rate * grad_c
d -= learning_rate * grad_d


print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

