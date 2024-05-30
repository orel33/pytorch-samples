# https://networkx.org/documentation/stable/reference/index.html
# https://networkx.org/documentation/stable/tutorial.html


import nxmetis
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def grid_layout(g):
    pos = {}
    for node in G.nodes():
        pos[node] = [node[1], node[0]]  # coord x (col), coord y (row)
    return pos


nrows = 3
ncols = 4
# WARNING: the grid dimension are swapped!
G = nx.grid_graph(dim=(ncols, nrows))
print("nodes:", G.nodes())

# Plot the graph
# pos = nx.spring_layout(G)
pos = grid_layout(G)
print("pos:", pos)
nx.draw(G, pos, with_labels=True)
plt.show()
