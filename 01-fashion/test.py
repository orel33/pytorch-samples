# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import mymodel
import matplotlib.pyplot as plt

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = mymodel.NeuralNetwork().to(device)
model.load_state_dict(torch.load("mymodel.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
X, y = next(iter(test_dataloader))
print(f"Shape of X: {X.shape} {X.dtype}")  # [ N, C, H, W ]
print(f"Shape of y: {y.shape} {y.dtype}")  # [ N ]

model.eval()
with torch.no_grad():
    X = X.to(device)
    pred = model(X)
    print(f"pred shape: {pred.shape}")
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# plot the image
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
img = X.squeeze()  # remove the input dimension of size 1
plt.imshow(img, cmap='gray')
plt.show()


# EOF
