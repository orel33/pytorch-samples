
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import os
import numpy as np
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# load CORA dataset
dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.

# MLP model


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


model = MLP(hidden_channels=16)
print(model)

## summary
# pip install torchsummary
# from torchsummary import summary
# summary(model, (1, 1433))

model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Define optimizer.


def train():
    model.train()  # Set the module in training mode.
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # compare input (model output) with target (data.y)
    loss.backward()     # Derive gradients.
    optimizer.step()    # Update parameters based on gradients.
    return loss


def test():
    model.eval()  # Set the module in evaluation mode.
    out = model(data.x)  # Perform a single forward pass.
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# Train model
loss = []
acc = []
for epoch in range(1, 201):
    loss.append(train().detach().numpy())
    acc.append(test())

### plot accuracy and loss
# import matplotlib.pyplot as plt
# plt.figure(figsize=(5, 3))
# plt.plot(range(1, 201), loss/max(loss), 'r', label='Training Loss')
# plt.plot(range(1, 201), acc, 'b', label='Test Accuracy')
# plt.title('Training Loss and Test Accuracy Over Epochs for MLP')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# Model accuracy post-training
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

# EOF
