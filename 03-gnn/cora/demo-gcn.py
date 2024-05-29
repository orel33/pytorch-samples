
# graph node classification
# GCN model vs MLP model

from torch_geometric.utils import degree
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import sys
import os
import numpy as np
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

EPOCHS = 300

# load CORA dataset
dataset = Planetoid(root='data/Planetoid', name='Cora')
# transform=NormalizeFeatures())

data = dataset[0]  # Get the first graph object (the only one in this dataset).

# try to aggregate node features from neighbors (for MLP model)
# deg = degree(data.edge_index[0], data.num_nodes)
# data.x0 = data.x.clone()
# data.x.zero_()
# for i in range(data.num_edges):
#     src = data.edge_index[0, i]
#     dst = data.edge_index[1, i]
#     data.x[src] += data.x0[dst]

# for i in range(data.num_nodes):
#     data.x[i] = data.x0[i] + data.x[i] / deg[i]

NormalizeFeatures()(data)
# print(data.x[0, :1000])
# print(data.x[0, :30])
# sys.exit(0)

# plot


def plot(lossvec, accvec):
    plt.figure(figsize=(5, 3))
    lossvec = [l/max(lossvec) for l in lossvec]  # normalize loss
    plt.plot(range(EPOCHS), lossvec, 'r', label='Loss')
    plt.plot(range(EPOCHS), accvec, 'b', label='Accuracy')
    plt.title('Loss and Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# MLP model


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x = self.lin1(data.x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


# GCN model
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        # self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, dataset.num_classes)
        # self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
        # self.conv2 = GATConv(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # Note that the non-linearity is not integrated in the conv calls, so we use ReLU here.
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# create model
# model = MLP(hidden_channels=16)
model = GCN(hidden_channels=16)
print(model)

# summary
# pip install torchsummary
# from torchsummary import summary
# summary(model, (1, 1433))

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
# Define optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()  # Set the module in training mode.
    optimizer.zero_grad()  # Clear gradients.
    out = model(data)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    # compare input (model output) with target (data.y)
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) # used for GCN model?
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()         # Derive gradients.
    optimizer.step()        # Update parameters based on gradients.
    lossval = loss.item()     # Derive Python number from loss tensor.
    print(f"Loss: {lossval:.4f}")
    return lossval


def test():
    model.eval()  # Set the module in evaluation mode.
    out = model(data)  # Perform a single forward pass.
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # Check against ground-truth labels.
    correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    acc = int(correct.sum()) / int(data.test_mask.sum())
    print(f"Accuracy: {acc:.4f}")
    return acc


# Train model
lossvec = []
accvec = []
for epoch in range(EPOCHS):
    lossvec.append(train())
    accvec.append(test())

# plot accuracy and loss
plot(lossvec, accvec)


# EOF
