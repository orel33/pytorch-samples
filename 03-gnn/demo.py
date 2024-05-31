# Code for the demo of the partitioning module, based on:
# * GAP: https://github.com/saurabhdash/GCN_Partitioning
# * Spectral by Gatti et al: https://github.com/alga-hopf/dl-spectral-graph-partitioning
# * DRL by Gatti et al: https://github.com/alga-hopf/drl-graph-partitioning

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import nxmetis
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

# Cpu by default
device = 'cpu'
print('Device:', device)
print('Pytorch version:', torch.__version__)
print('')

# Seeds
torch.manual_seed(12345)

######################## Grid Plot ########################


def grid_layout(g):
    pos = {}
    for node in g.nodes():
        pos[node] = [node[1], node[0]]  # coord x (col), coord y (row)
    return pos


def grid_plot(nrows, ncols, part):
    g = nx.grid_graph(dim=(ncols, nrows))  # networkx graph
    # pos = nx.spring_layout(g)
    pos = grid_layout(g)
    colors = np.array(['red', 'blue'])
    nx.draw(g, pos, node_color=colors[part], with_labels=True)
    plt.show()


######################## Data ########################


def grid_graph(nrows, ncols):
    # WARNING: the grid dimension are swapped!
    g = nx.grid_graph(dim=(ncols, nrows))  # networkx graph
    adj_sparse = nx.to_scipy_sparse_array(g, format='coo')
    row = adj_sparse.row
    col = adj_sparse.col
    rowcol = np.array([row, col])
    edges = torch.tensor(rowcol, dtype=torch.long)
    # append one hot encoding of the desired partition
    one_hot = []
    for node in g.nodes():
        i, j = node[0], node[1]
        # print("node:", node, "part:", j < ncols // 2)
        if j < ncols // 2:  # vertical cut
            one_hot.append([0., 1.])  # partition 0
        else:
            one_hot.append([1., 0.])  # partition 1
    nodes = torch.tensor(np.array(one_hot), dtype=torch.float)
    # append dummy node features
    # nodes = torch.ones((g.number_of_nodes(), 1), dtype=torch.float)
    # nodes = torch.rand((g.number_of_nodes(), 1), dtype=torch.float)
    G = Data(x=nodes, edge_index=edges)    # pytorch geometric graph
    return G

######################## Loss Function ########################

# Loss function for the partitioning module (pytorch graph)


def loss_normalized_cut(y_pred, graph):
    y = y_pred
    d = degree(graph.edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
    c = torch.sum(y[graph.edge_index[0], 0] * y[graph.edge_index[1], 1])
    return torch.sum(torch.div(c, gamma))

######################## Partitioning Module ########################

# Neural network for the partitioning module (bi-partitioning)


class ModelPartitioning(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels    # nb node features
        self.hidden_channels = hidden_channels
        self.output_channels = 2                # nb partitions
        self.activation = torch.tanh            # FIXME: use ReLU?
        self.conv0 = SAGEConv(self.input_channels, self.hidden_channels)
        self.conv1 = SAGEConv(self.hidden_channels, self.hidden_channels)
        self.conv2 = SAGEConv(self.hidden_channels, self.hidden_channels)
        self.lin0 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.lin1 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.final = nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.activation(self.conv0(x, edge_index))
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        # FIXME: use dropout?
        x = self.activation(self.lin0(x))
        x = self.activation(self.lin1(x))
        x = self.final(x)
        x = torch.softmax(x, dim=1)  # FIXME: use log_softmax ?
        return x


######################## Training the Partitioning Module ########################


model = ModelPartitioning(input_channels=2, hidden_channels=16).to(device)
print('Number of parameters:', sum(p.numel() for p in model.parameters()))
print('')

# Define the dataset with pyg graph as tensor
dataset = []
dataset.append(grid_graph(2, 2))
dataset.append(grid_graph(3, 4))
dataset.append(grid_graph(4, 4))
dataset.append(grid_graph(2, 6))
dataset.append(grid_graph(3, 6))
dataset.append(grid_graph(4, 6))
dataset.append(grid_graph(5, 6))
dataset.append(grid_graph(6, 6))
loader = DataLoader(dataset, batch_size=1, shuffle=True)

loss_fn = loss_normalized_cut
lr = 0.0001  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
epochs = 1000  # epochs
print_loss = 100  # steps after which the loss function is printed
losses = []


# epochs = 0
# Training loop
print('Start training')
for i in range(epochs):
    for d in loader:
        loss = torch.tensor(0.).to(device)
        d = d.to(device)
        data = model(d)
        loss = loss_fn(data, d)
        optimizer.zero_grad()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    if i % print_loss == 0:
        print('Epoch:', i, '   Loss:', loss.item())

print('End training')
print('')

# Test the model
nrows = ncols = 8
gg = grid_graph(nrows, ncols)
gg = gg.to(device)
print('Grid Graph', nrows, 'x', ncols)
print(gg)
data = model(gg)
part = torch.argmax(data, dim=1)
print('Model output:', data)
print('Partition:', part)
print('Loss:', loss_fn(data, gg).item())
print('')

# Plot output
grid_plot(nrows, ncols, part)
