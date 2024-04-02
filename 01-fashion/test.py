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
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
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


# print("test classes:", classes[[1, 2]])

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

N = 2
# X = torch.rand(N, 1, 28, 28, device=device)  # random images
test_dataloader = DataLoader(test_data, batch_size=N, shuffle=True)
X, y = next(iter(test_dataloader))  # get the first batch
# C=1 because the images are grayscale (1 channel)
print(f"Shape of X: {X.shape} {X.dtype}")  # [ N, C=1, H=28, W=28 ]
print(f"Shape of y: {y.shape} {y.dtype}")  # [ N ]
print(f"y: {y}")


model.eval()
with torch.no_grad():
    X = X.to(device)
    pred = model(X)  # [ N=2, C=10 ] (10 classes)
    # Nota Bene: pred are the logits, not the probabilities.
    print(f"pred shape: {pred.shape}")
    print(f"pred: {pred}")
    y_pred = pred.argmax(dim=1)
    # y_max = pred.max(dim=1)
    print(f"y_pred: {y_pred}")
    predicted = [classes[i] for i in y_pred]
    actual = [classes[i] for i in y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# plot the first image
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
img = X[0].squeeze()  # remove the input dimension of size 1
plt.imshow(img, cmap='gray')
plt.show()


# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")


# EOF
