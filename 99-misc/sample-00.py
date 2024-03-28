# -*- coding: utf-8 -*-
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import numpy as np
import math

# Create random input data (array of 2000 elements)
x = np.linspace(-math.pi, math.pi, 2000)
print(type(x))
print(x.shape)
print("x=", x)

# compute output data (array of 2000 elements)
y = np.sin(x)
print("y=", y)
