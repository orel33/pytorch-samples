# demo.py DNN with MNIST dataset and PyTorch

# Example of classification with a fully connected neural network, using Pytorch
# Objectives: Recognizing handwritten numbers ny using a classifier DNN network

# The MNIST dataset (Modified National Institute of Standards and Technology) is
# a must for Deep Learning. It consists of 60,000 small images of handwritten
# numbers for learning and 10,000 for testing.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision  # to get the MNIST dataset
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader

# get and format the training set
training_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

# get and format the test set
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

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


def fit(model, dataloader, epochs):

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(X)
            # X: 512x1x28x28 (float32)
            # y_pred_proba: 512x10 (float32)
            # y: 512 (int64)
            # y_pred_class = torch.argmax(y_pred, dim=1)
            curr_loss = loss(y_pred, y)
            curr_loss.backward()
            optimizer.step()
        print('epoch = ', epoch, 'loss =', curr_loss.item())


######################################################################
#                             TRAIN                                  #
######################################################################

EPOCHS = 5
BATCH_SIZE = 512
model = MyModel()
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

# first image and label in the training dataset
# X0, y0 = train_dataloader.dataset[0]
# print(f"X0: {X0.shape} (dtype: {X0.dtype})")
# print(f"y0: {y0}")

fit(model, train_dataloader, EPOCHS)

######################################################################
#                              TEST                                  #
######################################################################

# test the model on 1000 random images
NTESTS = 1000
test_dataloader = DataLoader(test_data, batch_size=NTESTS, shuffle=True)
X_test, y_test = next(iter(test_dataloader))  # get the first batch
y_test_pred = model(X_test)
loss = nn.CrossEntropyLoss()
curr_loss = loss(y_test_pred, y_test)
y_test_pred_class = torch.argmax(y_test_pred, dim=1)
accuracy = (y_test_pred_class == y_test).float().mean()

print('Test loss     :', curr_loss.item())
print('Test accuracy :', accuracy.item())

# EOF
