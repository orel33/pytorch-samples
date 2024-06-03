# <https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>

import torch
import numpy as np

# initialize a tensor directly from data (python list)
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# what is the difference between a tensor x and x.data?

print(f"Tensor: {x_data}\n")
print(f"Tensor Data: {x_data.data}\n")

# initialize a tensor from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Initialize a tensor with random or constant values. The shape is a tuple of
# tensor dimensions.

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
# same shape as ones_tensor, and random values
rand_tensor_like = torch.rand_like(ones_tensor)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")
print(f"Random Tensor Like: \n {rand_tensor_like} \n")

# Tensor attributes describe their shape, datatype, and the device on which they
# are stored.

# tensor = torch.randint(0, 10, (3, 4))   # integer random values in [0, 10)
tensor = torch.rand(3, 4)  # float random values in [0, 1]
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor converting to another datatype

tc = torch.rand(3, 4)*10
tc = tc.int()  # convert: .float() .double() .int() .long()
print(f"Datatype of tc: {tc.dtype}")
print(tc)


# Operations on Tensors: <https://pytorch.org/docs/stable/torch.html>

# Each of these operations can be run on the GPU (at typically higher speeds
# than on a CPU). If you’re using Colab, allocate a GPU by going to Runtime >
# Change runtime type > GPU.

# By default, tensors are created on the CPU. We need to explicitly move tensors
# to the GPU using ``.to`` method (after checking for GPU availability). Keep in
# mind that copying large tensors across devices can be expensive in terms of
# time and memory!

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Standard numpy-like indexing and slicing

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
# set all elements in the column 1 to zero
tensor[:, 1] = 0


# concatenate a sequence of tensors along a given dimension
t0 = torch.ones(4, 5)  # [4, 5] => 4 lines and 5 columns
print(t0)
t1 = torch.cat([t0, t0*2, t0*3], dim=1)  # [4, 3*5]
print(t1)
t2 = torch.cat([t0, t0*2, t0*3], dim=0)  # [3*4, 5]
print(t2)

# matrix multiplication (@) and transposition (tensor.T)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)  # equivalent to the previous one

print(f"tensor: \n {tensor} \n")
print(f"tensor.T: \n {tensor.T} \n")
print(f"tensor @ tensor.T: \n {tensor @ tensor.T} \n")

# element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)  # equivalent to the previous one

print(f"tensor * tensor.T: \n {tensor * tensor.T} \n")

# single-element tensor, shape []
agg = tensor.sum()
agg_item = agg.item()  # convert it to a Python numerical value
print(agg.shape, agg_item, type(agg_item))

# In-place operations, store the result into the operand (denoted by a _ suffix)

print(f"{tensor} \n")
tensor.add_(5).t_()
print(tensor)

# Nota Bene: In-place operations save some memory, but can be problematic when
# computing derivatives because of an immediate loss of history. Hence, their
# use is discouraged.

# Tensor to NumPy array (share same memory)
# A change in the tensor reflects in the NumPy array.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
# Changes in the NumPy array reflects in the tensor.
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# EOF
