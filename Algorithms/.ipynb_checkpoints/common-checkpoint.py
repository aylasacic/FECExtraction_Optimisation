import random
import pyzx as zx
import numpy as np
import fractions
import math
import networkx as nx
from functools import lru_cache

# full order for final pritnign
def full_order(order):
        return ["I"] + order + ["O"]

# cont edges corssing the cut
def boundary_size(G, selected):
    # graph cut size -> lower means the selection is more compact
    count = 0
    for u, w in G.edges():
        # exactly one endpoint is selected
        if (u in selected) != (w in selected):
            count += 1
    return count

# COPIED FROM SA CODE
def init_order(G, vertices):
    # pin I first, O last —> only optimise internal nodes (input and output do matter, even with OCM)
    internal = [v for v in vertices if v not in ("I", "O")]
    n_internal = len(internal)
    # if there a re no internal nodes, jsut return input and output, with no cut
    if n_internal == 0:
        return ["I", "O"], 0, []

    # start is the node with the maximum degree (if there is a tie, first one)
    # given the nodes and their degrees -> (ndoe, degree)
    # print(G.subgraph(internal).degree())
    # print(max(G.subgraph(internal).degree(), key = lambda x: x[1]))
    start = min(G.subgraph(internal).degree(), key = lambda x: x[1])[0]
    # print(start)
    # get the maximum degree (second entry in the touple)

    # EXPLORE DFS TOO ???
    # dfs_order = list(nx.dfs_tree(G.subgraph(internal), start).nodes())
    # visited = set(dfs_order)
    # for v in internal:
    #     if v not in visited:
    #         dfs_order.append(v)
    # return dfs_order, n_internal

    # get the initial order (BFS)
    bfs_order = list(nx.bfs_tree(G.subgraph(internal), start).nodes())
    # get the set of visited nodes
    visited = set(bfs_order)
    # if a ndoe is not visited, just append it
    # MAYBE RUN BFS AGAIN ON THE UNVISITED NOTES????
    for v in internal:
        if v not in visited:
            bfs_order.append(v)
    return bfs_order, n_internal

# full order for final pritnign
def full_order(order):
        return ["I"] + order + ["O"]

def compute_cut(order, G):
    # compute initial cut[i] = edges crossing between first (i+1) and remaining vertices

    n = len(order)
    # get {node/vertex: timestep}
    pos = {v: i for i, v in enumerate(order)}
    
    cut = [0] * (n - 1)

    # for each edge (between nodes u and v)
    for u, v, _ in G.edges(keys = True):
        # get their positons in the ordering (i.e.,  their current timestep)
        pu, pv = pos[u], pos[v]
        # if ord of u > ord v
        if pu > pv:
            # order them
            # ex. if (pu, pv) = (7,2)
            # make it (2,7) so the range works (positively) :D
            pu, pv = pv, pu
        # an edge contributes to every cut boundary between its two
        # endpoints
        # we use the position in the initial calculation
        for k in range(pu, pv):
            # for each node in between, add the edge to the edge count (i.e., add that edge to the cut)
            cut[k] += 1
    # print(pos) 
    boundaries = [(order[i], order[i + 1]) for i in range(n - 1)]

    return cut, boundaries

# ONLY LINEAR OREDING
# def compute_cut_fast_old(G, order):
#     """
#     faster version of compute_cut [I THINK]
#     - based mainly on C/C++ish logic
#     """

#     n = len(order)
#     pos = {v: i for i, v in enumerate(order)}

#     diff = [0] * n

#     edge_iter = G.edges(keys = True)
#     for u, v, _ in edge_iter:
#         # ignore self loops
#         if u == v:
#             continue

#         pu, pv = pos[u], pos[v]

#         if pu > pv:
#             pu, pv = pv, pu

#         diff[pu] += 1
#         diff[pv] -= 1

#     cut = []
#     running = 0

#     for i in range(n - 1):
#         running += diff[i]
#         cut.append(running)

#     boundaries = [(order[i], order[i + 1]) for i in range(n - 1)]
#     max_cut = max(cut) if cut else 0

#     return max_cut, cut, boundaries

def compute_cut_fast(G, schedule):
    # accept either:
    # order    = ["I", "A", "B", ...]
    # schedule = [["A", "B"], ["C"], ...]

    if schedule and not isinstance(schedule[0], (list, tuple, set)):
        order = list(schedule)
        pos = {v: i for i, v in enumerate(order)}
        n = len(order)
        # difference array that will be used to compute cut widths
        diff = [0] * n

        try:
            edge_iter = G.edges(keys = True)
        except TypeError:
            edge_iter = ((u, v, None) for u, v in G.edges())

        # for each set of vertices connected by same edge
        for u, v, _ in edge_iter:
            # ignore self-loops
            if u == v:
                continue

            # get the position of each node
            pu, pv = pos[u], pos[v]

            # switch the nodes so that they're ordered 
            # from smaller to larger
            if pu > pv:
                pu, pv = pv, pu

            # marks in the difference array that this edge starts contributing to the cut 
            # at position pu and stops contributing at position pv
            diff[pu] += 1
            diff[pv] -= 1

        cuts = []
        running = 0

        # iterate over every boundary between adjacent nodes (n-1 of them)
        for i in range(n - 1):
            # add the difference value at this position to the running total 
            # this gives the exact cut width at this boundary
            running += diff[i]
            cuts.append(running)

        # list of pairs of adjacent nodes
        boundaries = [(order[i], order[i + 1])  for i in range(n - 1)]

        return max(cuts), cuts, boundaries
    # fix schedule into a list
    schedule = [step for step in schedule if step]
    processed = {"I"}
    cuts = []

    # iterate over each timestamp
    for timestep in schedule:
        # add every qubit in this timetep into processed -> matk them as scheduled
        processed.update(timestep)
        # SAME AS BEFORE -> FRONTIEr
        # for every edge (pait of vertives) checks if exactly one endpoint is in processed
        # if so, the edge crosses the boundary and counts as 1 
        # summing all of these gives the total cut width at this moment
        # COULD BE FASTER MAYVE????
        cut = sum((u in processed) != (v in processed) for u, v in G.edges())
        cuts.append(cut)

    boundaries = [tuple(timestep) for timestep in schedule]

    return max(cuts, default=0), cuts, boundaries