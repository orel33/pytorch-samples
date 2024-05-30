# pip install networkx-metis
# https://networkx-metis.readthedocs.io/en/latest/reference/nxmetis.html

import nxmetis
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def grid_layout(g):
    pos = {}
    for node in G.nodes():
        pos[node] = [node[0], node[1]]
    # print("pos:", pos)
    return pos


G = nx.grid_graph(dim=(3, 4))
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


# compute colors
color = np.array(['r', 'b'])
parts = []
for node in G.nodes():
    if node in lparts[0]:
        parts.append(0)
    else:
        parts.append(1)

print(parts)
nx.draw(G, pos, node_color=color[parts], with_labels=True)
plt.show()
