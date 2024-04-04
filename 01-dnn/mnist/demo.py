# demo.py DNN with MNIST dataset and PyTorch

# Example of classification with a fully connected neural network, using Pytorch
# Objectives: Recognizing handwritten numbers ny using a classifier DNN network

# The MNIST dataset (Modified National Institute of Standards and Technology) is
# a must for Deep Learning. It consists of 60,000 small images of handwritten
# numbers for learning and 10,000 for testing.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision  # to get the MNIST dataset
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# get and format the training set
mnist_trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=None)
x_train = mnist_trainset.data.type(torch.DoubleTensor)  # 60000x28x28
y_train = mnist_trainset.targets  # 60000

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

# get and format the test set
mnist_testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=None)
x_test = mnist_testset.data.type(torch.DoubleTensor)  # 10000x28x28
y_test = mnist_testset.targets  # 10000

# Normalize the data
print('Before normalization : Min={}, max={}'.format(
    x_train.min(), x_train.max()))

xmax = x_train.max()
x_train = x_train / xmax
x_test = x_test / xmax

print('After normalization  : Min={}, max={}'.format(
    x_train.min(), x_train.max()))


# Model definition
class MyModel(nn.Module):
    """
    Basic fully connected neural-network
    """

    def __init__(self):
        hidden1 = 100
        hidden2 = 100
        super(MyModel, self).__init__()
        self.hidden1 = nn.Linear(28*28, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, 10)

    def forward(self, x):
        # flatten the images before using fully-connected layers
        x = x.view(-1, 28*28)  # the size -1 is inferred from other dimensions
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.softmax(x, dim=0)
        return x


def fit(model, X_train, Y_train, X_test, Y_test, EPOCHS=5, BATCH_SIZE=32):

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)  # lr is the learning rate
    model.train()

    # history = convergence_history_CrossEntropyLoss()
    # history.update(model, X_train, Y_train, X_test, Y_test)

    n = X_train.shape[0]  # number of observations in the training data

    # stochastic gradient descent
    for epoch in range(EPOCHS):

        print('Epoch', epoch+1, '/', EPOCHS)

        batch_start = 0
        epoch_shuffler = np.arange(n)
        # remark that 'utilsData.DataLoader' could be used instead
        np.random.shuffle(epoch_shuffler)

        while batch_start+BATCH_SIZE < n:
            # get mini-batch observation
            mini_batch_observations = epoch_shuffler[batch_start:batch_start+BATCH_SIZE]
            # the input image is flattened
            var_X_batch = Variable(
                X_train[mini_batch_observations, :, :]).float()
            var_Y_batch = Variable(Y_train[mini_batch_observations])

            # gradient descent step
            optimizer.zero_grad()  # set the parameters gradients to 0
            # predict y with the current NN parameters
            Y_pred_batch = model(var_X_batch)

            # compute the current loss
            curr_loss = loss(Y_pred_batch, var_Y_batch)
            curr_loss.backward()  # compute the loss gradient w.r.t. all NN parameters
            optimizer.step()  # update the NN parameters

            # prepare the next mini-batch of the epoch
            batch_start += BATCH_SIZE

    #     history.update(model, X_train, Y_train, X_test, Y_test)

    # return history


######################################################################
#                             MAIN                                   #
######################################################################

model = MyModel()


batch_size = 512
epochs = 128

fit(model, x_train, y_train, x_test, y_test,
    EPOCHS=epochs, BATCH_SIZE=batch_size)


var_x_test = Variable(x_test[:, :, :]).float()
var_y_test = Variable(y_test[:])
y_pred = model(var_x_test)

loss = nn.CrossEntropyLoss()
curr_loss = loss(y_pred, var_y_test)

val_loss = curr_loss.item()
val_accuracy = float(
    (torch.argmax(y_pred, dim=1) == var_y_test).float().mean())

print('Test loss     :', val_loss)
print('Test accuracy :', val_accuracy)

# EOF
