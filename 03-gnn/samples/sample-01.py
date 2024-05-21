################## CORA dataset ##################

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object (the only one in this dataset).

print()
print(data)
print()
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of test nodes: {data.test_mask.sum()}')
print(f'Number of val nodes: {data.val_mask.sum()}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print()
print('===========================================================================================================')
print("train mask:", data.train_mask, data.train_mask.sum())
print("test mask:", data.test_mask, data.test_mask.sum())
print("val mask:", data.val_mask, data.val_mask.sum())

print()
print('===========================================================================================================')
print("label y:", data.y, "min:", data.y.min().item(), "max:", data.y.max().item())
print("features x:", data.x.shape)

# loop over the edges in the graph

train_halo = []
for i in range(data.num_edges):
    # src, dst = data.edge_index[:, i]
    src = data.edge_index[0, i]
    dst = data.edge_index[1, i]
    # print(f'Edge {i}: {src.item()} -> {dst.item()}')
    if data.train_mask[src] and not data.train_mask[dst]:
        train_halo.append(dst.item())

print("halo size:", len(train_halo))
