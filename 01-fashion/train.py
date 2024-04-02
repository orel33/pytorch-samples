# demo.py

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import mymodel

# Get cpu, gpu or mps device for training.
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print("train data size: ", len(training_data))
print("test data size: ", len(test_data))


# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# train_size = len(train_dataloader.dataset)
# test_size = len(test_dataloader.dataset)
# print("train dataset size: ", train_size)
# print("test dataset size: ", test_size)

# first image and label in the training dataset
# X0, y0 = train_dataloader.dataset[0]
# print(f"Shape of X0: {X0.shape}")
# print(f"y0: {y0}")

# Iterate through the dataloader once
# N: batch size, C: color channels, H: height, W: width
# for X, y in train_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

nb_batches = len(train_dataloader)
print(f"Number of batches: {nb_batches}, batch size: {batch_size}")

for batch, (X, y) in enumerate(train_dataloader):
    print(f"batch index: {batch}")
    print(f"Shape of X: {X.shape} {X.dtype}")  # [N, C, H, W]
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# create the model
model = mymodel.NeuralNetwork().to(device)
print(model)

# training the model
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# size = len(train_dataloader.dataset)
model.train()
# get first batch of shape X=[N, C, H, W], y=[N]
X, y = next(iter(train_dataloader))
print(f"Shape of X: {X.shape} {X.dtype}")
print(f"Shape of y: {y.shape} {y.dtype}")
pred = model(X)  # forward pass (output of shape [N, 10])
print(f"Shape of prediction for X0: {pred.shape} {pred.dtype}")
# print(f"prediction for X0: {pred0}")
loss = loss_fn(pred, y)
print(f"Loss: {loss}, shape: {loss.shape}, type: {loss.dtype}")
print(f"Loss: {loss}")
# loss.backward()   # back-propagation
# optimizer.step()  # update weights
# optimizer.zero_grad()  # zero the gradients
# pred = model(X)  # forward pass (output of shape [N, 10])
# loss = loss_fn(pred, y)


######################################################################
# Training Loop
######################################################################


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    mymodel.train(train_dataloader, model, loss_fn, optimizer, device)
    mymodel.test(test_dataloader, model, loss_fn, device)
print("Done!")


# Save the model to disk
torch.save(model.state_dict(), "mymodel.pth")
print("Saved PyTorch Model State to model.pth")


# EOF
