import networkx as nx
import nxmetis
import timeit
import torch
from torch_geometric.utils import from_networkx, degree, normalized_cut

# convert partition from networkx to list

# nxpart: [ [list of nodes in partition 0], [list of nodes in partition 1] ]
# nxpart is the output of nxmetis.partition

# part: [ partition of node 0, partition of node 1, ...] as a list

# edge_index: [ [source nodes], [target nodes] ] as a list of lists


def nxpart2list(g, nxpart):
    dictpart = {}
    for p, nodes in enumerate(nxpart):
        for n in nodes:
            dictpart[n] = p
    part = [dictpart[n] for n in g.nodes()]
    return part

# cut for networkx graph


def nx_cut(g, part):
    cut = 0
    for u, v in g.edges():
        print("u:", u, "v:", v)
        if part[u] != part[v]:
            cut += 1
    return cut / 2


# normalized cut for networkx graph

def nx_ncut(g, part):
    cut = nx_cut(g, part)
    degA = sum(g.degree(n) for n in part[0])
    degB = sum(g.degree(n) for n in part[1])
    return cut * (1 / degA + 1 / degB)

# loss function for normalized cut (for pytorch geometric graph)


def pyg_loss_ncut(pyg_graph, y_pred):
    y = y_pred
    d = degree(pyg_graph.edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
    c = torch.sum(y[pyg_graph.edge_index[0], 0]
                  * y[pyg_graph.edge_index[1], 1])
    return torch.sum(torch.div(c, gamma))


# def pyg_ncut(pyg_graph, pyg_part):
#     ncut = normalized_cut(pyg_graph.edge_index, pyg_part)
#     return ncut

# cut for pytorch geometric graph
# part must be a tensor of size (nb nodes)


def pyg_cut(pyg_graph, pyg_part):
    cut = torch.sum(pyg_part[pyg_graph.edge_index[0, :]] !=
                    pyg_part[pyg_graph.edge_index[1, :]]).item() / 2
    return cut


########################### main ###########################
nrows = ncols = 10
g = nx.grid_graph(dim=(ncols, nrows))  # networkx graph
t0 = timeit.default_timer()
mcut, nxpart = nxmetis.partition(g, 2)
t1 = timeit.default_timer() - t0
print("METIS time:", t1)
print("METIS cut:", mcut)
cut = nx.cut_size(g, nxpart[0], nxpart[1])
ncut = nx.normalized_cut_size(g, nxpart[0], nxpart[1])
print("nx cut:", cut)
print("nx ncut:", ncut)


# convert to pytorch geometric graph
pyg_graph = from_networkx(g)
part = nxpart2list(g, nxpart)
print("METIS part:", part)
pyg_part = torch.tensor(part, dtype=torch.long)

# cut2 = nx_cut(g, part)
# ncut2 = nx_ncut(g, part)
# print("cut2:", cut2)
# print("ncut2:", ncut2)

pygcut = pyg_cut(pyg_graph, pyg_part)
print("pyg cut:", pygcut)
# pygncut = pyg_ncut(pyg_graph, pyg_part)
# print("pyg ncut:", pygncut)

# EOF
