# https://scikit-network.readthedocs.io/en/latest/first_steps.html#installation
# pip install "numpy>=1.17.3,<1.25.0"
# pip install scikit-network
# https://scikit-network.readthedocs.io/en/latest/tutorials/embedding/pca.html

import matplotlib.pyplot as plt
from sknetwork.visualization import visualize_graph
from sknetwork.embedding import PCA
from sknetwork.data import karate_club
import numpy as np
import scipy as sp
from sknetwork.data import grid
import networkx as nx
import matplotlib.pylab as plt

################ karate club ################

graph = karate_club(metadata=True)
# print(graph)
adjacency = graph.adjacency
position = graph.position
labels = graph.labels
#print(labels)
nxgraph = nx.from_scipy_sparse_array(adjacency)
pca = PCA(2)
embedding = pca.fit_transform(adjacency)    # embedding in 2 dimensions => useful for 2d visualization
# colors = np.array(['red', 'blue'])
# nx.draw(nxgraph, position,  with_labels=True)
# nx.draw(nxgraph, embedding, node_color=colors[labels], with_labels=True)
# plt.show()

# visualization as SVG image 
# from IPython.display import SVG, display
# image = visualize_graph(adjacency, position, labels=labels)  # as 'str' type
# SVG(image)

################ grid ################

graph = grid(10, 5, metadata=True)
position = graph.position
adjacency = graph.adjacency # of type 'scipy.sparse._csr.csr_matrix'
nxgraph = nx.from_scipy_sparse_array(adjacency)
pca = PCA(2)
embedding = pca.fit_transform(adjacency)    # embedding in 2 dimensions => useful for 2d visualization
# nx.draw(nxgraph, position, with_labels=True)
nx.draw(nxgraph, embedding, with_labels=True)
plt.show()

# EOF
