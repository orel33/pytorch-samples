# metis module: https://metis.readthedocs.io/en/latest/

import metis
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()
G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)

# Partition the graph
(edgecut, parts) = metis.part_graph(G, 2)

# Print the edgecuts
print(edgecut)

# Print the parts
print(parts)

# Plot the graph
pos = nx.spring_layout(G)
color = np.array(['r', 'b'])
# print(color, type(color))
# node_color = color[parts]
# print(node_color, type(node_color))
nx.draw(G, pos, node_color=color[parts], with_labels=True)
plt.show()
