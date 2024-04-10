import torch

# ---- First Example

a = torch.IntTensor([1, 2, 3, 4, 5])
# a = torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("a:", a)
mask = torch.BoolTensor([0, 1, 1, 0, 1])
print("mask:", mask)
# result = torch.masked_select(a, mask)
result = a[mask]
print("result:", result)

# ---- Mask
#
# A mask is a tensor that has the same shape as the input tensor and contains
# binary values. The values in the mask tensor are used to filter out elements
# from the input tensor. The mask tensor is multiplied element-wise with the
# input tensor, and only the elements that correspond to a value of 1 in the
# mask tensor are kept.

# ---- Example
#
# Let's see how to use a mask tensor to filter out elements from an input tensor.
# We will create a mask tensor with the same shape as the input tensor and
# filter out all the elements that are less than 3.

# create a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("x:", x)
# create a mask tensor
mask = x.ge(3)
print("mask:", mask)
# apply the mask to the tensor
result = x[mask]
print("result:", result)
