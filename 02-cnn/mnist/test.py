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

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

print(test_data)

################################################

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1)

################################################
# Testing the model
################################################


def test():
    cnn.eval()
    with torch.no_grad():
        correct = 0     # FIXME: not used
        total = 0
        for images, labels in test_loader:
            test_output = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))

        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


################################################
# Main
################################################

test()

# => Test Accuracy of the model on the 10000 test images: 0.99

################################################
