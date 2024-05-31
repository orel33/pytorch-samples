# pip install networkx-metis
# https://networkx-metis.readthedocs.io/en/latest/reference/nxmetis.html

import nxmetis
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def grid_layout(g):
    pos = {}
    for node in g.nodes():
        pos[node] = [node[1], node[0]]  # coord x, coord y
    # print("pos:", pos)
    return pos


nrows = 3
ncols = 4
# WARNING: the grid dimension are swapped!
G = nx.grid_graph(dim=(ncols, nrows))
print("nodes:", G.nodes())  # hashable items

edgecut, lparts = nxmetis.partition(G, 2)

# Print the edgecuts
print(edgecut)

# Print the parts
print("part 0:", lparts[0])
print("part 1:", lparts[1])

# Plot the graph
# pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
pos = grid_layout(G)

dictpart = {}
for p, nodes in enumerate(lparts):
    for n in nodes:
        dictpart[n] = p

parts = [dictpart[n] for n in G.nodes()]
print(parts)

colors = np.array(['red', 'blue', 'green'])
nx.draw(G, pos, node_color=colors[parts], with_labels=True)
plt.show()
