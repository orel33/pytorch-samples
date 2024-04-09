# https: //medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision import datasets


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################
# Loading the dataset
################################################

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)

print(train_data)
print(train_data.data.size())
print(train_data.targets.size())

################################################

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=1)

################################################
# Displaying images
################################################

# Displaying the first image
# plt.imshow(train_data.data[0], cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()


def plot_images(data, cols, rows):
    figure = plt.figure(figsize=(10, 8))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# plot_images(train_data, 5, 5)

################################################
# Defining the model
################################################

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


################################################
# Training the model
################################################

def train(num_epochs, cnn, loader):

    cnn.train()
    total_step = len(loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):
            # gives batch data, normalize x when iterate train_loader
            b_x = images   # batch x
            b_y = labels   # batch y
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()   # clear gradients
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

################################################
# Main
################################################


cnn = CNN().to(device)
print(cnn)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
num_epochs = 10
train(num_epochs, cnn, train_loader)
cnn.save('model.pth')

################################################
