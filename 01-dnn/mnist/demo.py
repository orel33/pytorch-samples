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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Nota Bene: transform is a function that takes in an PIL image and returns a
# transformed version. For instance, ToTensor() converts a PIL image or
# numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of
# shape (C x H x W) in the range [0.0, 1.0].
# https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
# C: color channels (in our case, C=1 because we have grayscale images)

# get and format the training set
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

# get and format the test set
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

# # get first image and label in the training dataset
# X0, y0 = training_dataset[0]
# print(f"Training set X0: {X0.shape} ({X0.dtype})")  # tensor 1x28x28 (float32)
# print(f"Training set y0: {y0} ({type(y0)})")        # scalar (int)
# # print(f"first image: {X0.flatten()}")


# # get raw 'data' and 'targets' from the training set
# # Tips: it gives better performance to use the raw data and normalize it by hand
XMAX = 255
X_train = train_dataset.data.type(torch.FloatTensor) / XMAX
y_train = train_dataset.targets                       # raw targets
# print(f"Training set X: {X_train.shape} {X_train.dtype}")
# print(f"Training set y: {y_train.shape} {y_train.dtype}")
# # print(f"first image: {X_train[0].flatten()}")


######################################################################
#                             MODEL                                  #
######################################################################


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

######################################################################
#                            MY DATALOADER                           #
######################################################################

# Get a batch of images (X) and labels (y)


class MyDataLoader():

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # normalize the image data in range [0,1]
        XMAX = 255
        self.X = dataset.data.type(torch.FloatTensor) / XMAX
        self.y = dataset.targets
        self.idx = 0

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        start = self.idx * self.batch_size
        stop = start + self.batch_size
        X_batch = self.X[start:stop, ...]  # 512x28x28
        y_batch = self.y[start:stop]       # 512
        self.idx += 1
        return X_batch, y_batch


######################################################################
#                             TRAIN                                  #
######################################################################


def train(model, dataloader, epochs, batch_size):

    count = 0
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    nb_batches = len(dataloader)
    print(f"nb batches: {nb_batches} (batch size: {batch_size})")

    for epoch in range(epochs):
        start = 0
        for X, y in dataloader:
            # print(f"X: {X.shape} {X.dtype}")
            # print(f"y: {y.shape} {y.dtype}")
            X, y = X.to(DEVICE), y.to(DEVICE)
            # X: 512x1x28x28 (float32) -> images
            # y: 512 (int64) -> classes
            optimizer.zero_grad()
            y_pred = model(X)
            # y_pred: 512x10 (float32) -> probabilities for each class
            count += len(y_pred)    # count the number of images processed
            curr_loss = loss(y_pred, y)
            curr_loss.backward()
            optimizer.step()
            start += BATCH_SIZE
        print(f'epoch: {epoch+1} / {epochs}, loss: {curr_loss.item()}')
    print('count:', count)


######################################################################
#                              MAIN                                  #
######################################################################

EPOCHS = 5
BATCH_SIZE = 512

model = MyModel().to(DEVICE)

# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE) # FIXME: very slow!!!
train_dataloader = MyDataLoader(train_dataset, batch_size=BATCH_SIZE)

# first image and label in the training dataset
# X0, y0 = train_dataloader.dataset[0]
# print(f"X0: {X0.shape} (dtype: {X0.dtype})")
# print(f"y0: {y0}")

print(f"Training on {DEVICE}")
train(model, train_dataloader, EPOCHS, BATCH_SIZE)


######################################################################
#                              TEST                                  #
######################################################################

# test the model on 1000 random images
NTESTS = 1000
print(f"Testing the model on {NTESTS} images")
test_dataloader = DataLoader(test_dataset, batch_size=NTESTS, shuffle=True)
X_test, y_test = next(iter(test_dataloader))  # get the first batch
model.eval()
y_test_pred = model(X_test)
loss = nn.CrossEntropyLoss()
curr_loss = loss(y_test_pred, y_test)
y_test_pred_class = torch.argmax(y_test_pred, dim=1)
accuracy = (y_test_pred_class == y_test).float().mean()

print('Test loss     :', curr_loss.item())
print('Test accuracy :', accuracy.item())

# EOF
