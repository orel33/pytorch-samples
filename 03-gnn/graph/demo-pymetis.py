# pip install pymetis
# https://documen.tician.de/pymetis/functionality.html


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pymetis

adjacency_list = [np.array([4, 2, 1]),
                  np.array([0, 2, 3]),
                  np.array([4, 3, 1, 0]),
                  np.array([1, 2, 5, 6]),
                  np.array([0, 2, 5]),
                  np.array([4, 3, 6]),
                  np.array([5, 3])]
edgecut, parts = pymetis.part_graph(2, adjacency=adjacency_list)
# edgecut = 3
# parts = [1, 1, 1, 0, 1, 0, 0]

# nodes_part_0 = np.argwhere(np.array(parts) == 0).ravel()  # [3, 5, 6]
# nodes_part_1 = np.argwhere(np.array(parts) == 1).ravel()  # [0, 1, 2, 4]

# Print the edgecuts
print(edgecut)

# Print the parts
print(parts)

# Plot the graph
G = nx.Graph()

pos = nx.spring_layout(G)
color = np.array(['r', 'b'])
nx.draw(G, pos, node_color=color[parts], with_labels=True)
plt.show()
