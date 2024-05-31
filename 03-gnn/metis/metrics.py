import networkx as nx
import nxmetis
import timeit
import torch

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

# normalized cut for pytorch geometric graph


def pyg_ncut(y_pred, graph):
    y = y_pred
    d = degree(graph.edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
    c = torch.sum(y[graph.edge_index[0], 0] * y[graph.edge_index[1], 1])
    return torch.sum(torch.div(c, gamma))

# cut for pytorch geometric graph


# def pyg_cut(graph):
#     cut = torch.sum((graph.x[graph.edge_index[0], :2] != graph.x[graph.edge_index[1], :2]).all(
#         axis=-1)).detach().item() / 2
#     return cut

def pyg_cut(g, part):
    cut = torch.sum(part[g.edge_index[0, :]] !=
                    part[g.edge_index[1, :]]).item() / 2
    return cut


########################### main ###########################
nrows = ncols = 10
g = nx.grid_graph(dim=(ncols, nrows))  # networkx graph
t0 = timeit.default_timer()
mcut, nxpart = nxmetis.partition(g, 2)
t1 = timeit.default_timer() - t0
print("METIS time:", t1)
print("METIS cut:", mcut)
part = nxpart2list(g, nxpart)
print("METIS part:", part)
# cut = nx_cut(g, part)
# ncut = nx_ncut(g, part)
cut = nx.cut_size(g, nxpart[0], nxpart[1])
print("cut:", cut)
ncut = nx.normalized_cut_size(g, nxpart[0], nxpart[1])
print("ncut:", ncut)

# EOF
