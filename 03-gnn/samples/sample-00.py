# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)


print()
print("data:", data)
print("edge index:", data.edge_index)
print("node features x:", data.x)

print('===================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


# loop over the edges in the graph

for i in range(data.num_edges):
    # src, dst = data.edge_index[:, i]
    src = data.edge_index[0, i]
    dst = data.edge_index[1, i]
    print(f'Edge {i}: {src.item()} -> {dst.item()}')
