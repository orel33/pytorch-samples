# https://networkx.org/documentation/stable/reference/index.html
# https://networkx.org/documentation/stable/tutorial.html


import nxmetis
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

G = nx.grid_graph(dim=(3, 4))
# G = nx.grid_graph(dim=(range(7, 9), range(3, 6)))

# Plot the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()
