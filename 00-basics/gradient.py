# https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/blob/master/Misc/

import torch

# ---- My basic function f


def f(x):
    y = x*x + 4*x - 5
    return y


def df(x):
    y = 2*x + 4
    return y


# ---- Examples :
print('f(3) is : ', f(3))
print('df(3) is : ', df(3))

# ---- Gradient

# define a tensor x with requires_grad=True

x = torch.tensor(3.0, requires_grad=True)
print("x:", x)
y = x*x + 4*x - 5
print("y:", y)
# compute the gradient of y with respect to x (dy/dx)
y.backward()
print("dy/dx=x.grad:", x.grad)

# By default, gradients are accumulated in buffers (i.e, not overwritten)
# whenever .backward() is called.

# ---- Another example
x.grad.zero_()  # reset the gradient
u = torch.tensor(1.0, requires_grad=True)
z = x*x + 2*x + x*u + 1
print("z:", z)
z.backward()  # compute the gradient of z with respect to x and u
print("dz/dx=x.grad:", x.grad)  # dz/dx = 2*x + 2 + u
print("dz/du=u.grad:", u.grad)  # dz/du = x
