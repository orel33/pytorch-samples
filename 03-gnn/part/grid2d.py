# generate grid graph

import torch
from torch_geometric.data import Data

print('============= grid 2d ==================')


def make_grid2d(height, width):
    edge_index = []
    for i in range(height):
        for j in range(width):
            if j + 1 < width:
                edge_index.append([i * width + j, i * width + j + 1])
                edge_index.append([i * width + j + 1, i * width + j])
            if i + 1 < height:
                edge_index.append([i * width + j, (i + 1) * width + j])
                edge_index.append([(i + 1) * width + j, i * width + j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index)


data = make_grid2d(3, 3)
# print(data.edge_index)
print(data)
