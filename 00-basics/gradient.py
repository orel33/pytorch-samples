# => https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle/-/blob/master/Misc/
# => https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward

import torch
import torch.nn.functional as F

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

# When we create a tensor, requires_grad is set to False by default, meaning it
# won't track operations. However, if we set requires_grad=True, PyTorch will
# start to track all operations on the tensor.

# define a tensor x with requires_grad=True

x = torch.tensor(3.0, requires_grad=True)
print("x:", x)
y = x*x + 4*x - 5
print("y:", y)
print("y.grad_fn:", y.grad_fn)
print("y.data:", y.data)
# compute the gradient of y with respect to x (dy/dx)
y.backward()
print("dy/dx=x.grad:", x.grad)

# By default, gradients are accumulated in buffers (i.e, not overwritten)
# whenever .backward() is called.

# ---- Another example
# x.grad.zero_()  # reset the gradient
x = torch.tensor(3.0, requires_grad=True)
u = torch.tensor(1.0, requires_grad=True)
z = x*x + 2*x + x*u + 1
print("z:", z)
# retain_graph=True to be able to call backward again, else the computational graph is deleted...
# z.backward(retain_graph=True)
# print("dz/dx=x.grad:", x.grad)  # dz/dx = 2*x + 2 + u
# print("dz/du=u.grad:", u.grad)  # dz/du = x
z.backward()  # compute the gradient of z with respect to x and u
print("dz/dx=x.grad:", x.grad)  # dz/dx = 2*x + 2 + u
print("dz/du=u.grad:", u.grad)  # dz/du = x

# ---- Tensor and gradient

# z.backward() is equivalent to z.backward(torch.tensor(1.0))
# z.backward(torch.tensor(1.0))

# When dealing with non-scalar tensors, backward requires an additional
# argument: the gradient of the tensor with respect to some scalar (usually a
# loss).

# By default, this gradient is a tensor with ones at the same shape as the
# tensor. But if you chain some layers manually, you can provide a custom
# gradient to the backward method.

# ---- Tensor and gradient

x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
print("x:", x)
y = x * x
zero = torch.zeros(2, 2)
# convert tensor result into a scalar loss
loss = F.mse_loss(y, zero)
# loss = y.sum()
print("y:", y)
print("loss:", loss)
loss.backward()
print("x.grad:", x.grad)

# ---- Example with a loss function and tensors

# Let's see how to use the gradient to minimize a loss function. We will use
# the loss function f(x) = x^2 + 4x - 5 and try to find the value of x that
# minimizes f(x).

# We can use the gradient to find the minimum of a function by iteratively
# moving in the opposite direction of the gradient. This is known as gradient
# descent.

# Let's define a learning rate and an initial value for x.

learning_rate = 0.15
x = torch.tensor(3.0, requires_grad=True)

# We can now run the gradient descent algorithm to minimize the loss function.

for i in range(100):
    y = f(x)
    y.backward()
    x.data = x.data - learning_rate * x.grad    # update x using gradient descent
    x.grad.zero_()
    if i % 10 == 0:
        print("i:", i, "x:", x.item())

# It converges to the minimum of the function f(), that is x = -2.

# After running the gradient descent algorithm for 100 iterations, we can see
# that the value of x converges to the minimum of the loss function.
