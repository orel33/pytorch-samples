import networkx as nx
import nxmetis
import timeit


# Returns the normalized cut, runtime, volumes and cut of the METIS
# partitioning of the networkx graph g


def normalized_cut_metis(g):
    t0 = timeit.default_timer()
    cut, parts = nxmetis.partition(g, 2)
    t1 = timeit.default_timer() - t0
    degA = sum(g.degree(n) for n in parts[0])
    degB = sum(g.degree(n) for n in parts[1])
    return cut * (1 / degA + 1 / degB), t1, degA, degB, cut


# TODO: try it
