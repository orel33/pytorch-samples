
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch_geometric.data import Data
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# Install required packages.
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Helper function for visualization.
# %matplotlib inline


def visualize(h, color, title):
    z = TSNE(n_components=2).fit_transform(
        h.detach().cpu().numpy())  # Dimension reduction algorithm
    palette = ["#9b5fe0", "#16a4d8", "#60dbe8",
               "#8bd346", "#efdf48", "#f9a52c", "#d64e12"]
    colors = [palette[color.detach().numpy()[i]] for i in range(len(color))]

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=colors)
    plt.title(title)
    plt.legend()
    plt.show()


# load CORA dataset

dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.

visualize(data.x, color=data.y, title='Space Embedding of CORA dataset')

# EOF
