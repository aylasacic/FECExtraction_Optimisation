import networkx as nx
from tqdm.auto import tqdm

def held_karp_cutwidth(G, first=None, last=None):
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return 0, nodes[:]
    if n > 26:
        raise ValueError(f"n = {n} is too large for an exact 2^n DP")

    idx = {u: i for i, u in enumerate(nodes)}

    # weighted adjecency graph
    w = [dict() for _ in range(n)]
    degw = [0] * n
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        if i == j:
            continue
        w[i][j] = w[i].get(j, 0) + 1
        w[j][i] = w[j].get(i, 0) + 1
        degw[i] += 1
        degw[j] += 1
        
    nbr = [[(1 << j, c) for j, c in w[i].items()] for i in range(n)]
    size = 1 << n
    full = size - 1
    INF = float("inf")

    # delta[S] = # edges (with multiplicity) crossing the S | (V\S) boundary
    delta = [0] * size
    for S in tqdm(range(1, size), desc = "Computing delta", leave = False, dynamic_ncols = True):
        v = (S & -S).bit_length() - 1
        prev = S ^ (1 << v)
        cross = 0
        for bit, c in nbr[v]:
            if prev & bit:
                cross += c
        delta[S] = delta[prev] + degw[v] - 2 * cross

    # optional endpoint pinning
    fbit = (1 << idx[first]) if first is not None else 0
    lbit = (1 << idx[last]) if last is not None else 0

    def valid(S):
        if first is not None and not (S & fbit):
            return False
        if last is not None and (S & lbit) and S != full:
            return False
        return True

    f = [0] * size
    parent = [-1] * size
    
    for S in tqdm(range(1, size), desc="Held-Karp DP"):
        if not valid(S):
            f[S] = INF
            continue
        best, best_v = INF, -1
        s = S
        while s:
            v = (s & -s).bit_length() - 1
            val = f[S ^ (1 << v)]
            if val < best:
                best, best_v = val, v
            s &= s - 1
        f[S] = max(delta[S], best)
        parent[S] = best_v

    # reconstruct the ordering (left -> right)
    order, S = [], full
    while S:
        v = parent[S]
        order.append(v)
        S ^= 1 << v
    order.reverse()
    return f[full], [nodes[i] for i in order]

# def held_karp_cutwidth(G, first = None, last = None):
#     """
#     Exact cutwidth + optimal vertex ordering via Held-Karp-style subset DP

#     first / last: pin a node to the leftmost / rightmost position -> fixed endbags (VV)
#     returns (width, order) with order in ORIGINAL labels
#     time O(n^2*2^n), 
#     space O(2^n). 
#     worked to ~n <= 25.
#     """
#     nodes = list(G.nodes())
#     n = len(nodes)
#     if n <= 1:
#         return 0, nodes[:]
#     if n > 26:
#         raise ValueError(f"n={n} is too large for an exact 2^n DP")

#     idx = {u: i for i, u in enumerate(nodes)}

#     # weighted adjecency graph
#     w = [dict() for _ in range(n)]
#     degw = [0] * n
#     for u, v in G.edges():
#         i, j = idx[u], idx[v]
#         if i == j:
#             continue
#         w[i][j] = w[i].get(j, 0) + 1
#         w[j][i] = w[j].get(i, 0) + 1
#         degw[i] += 1
#         degw[j] += 1
#     nbr = [[(1 << j, c) for j, c in w[i].items()] for i in range(n)]

#     size = 1 << n
#     full = size - 1
#     INF = float("inf")

#     # delta[S] = # edges (with multiplicity) crossing the S | (V\S) boundary
#     delta = [0] * size
#     for S in range(1, size):
#         v = (S & -S).bit_length() - 1
#         prev = S ^ (1 << v)
#         cross = 0
#         for bit, c in nbr[v]:
#             if prev & bit:
#                 cross += c
#         delta[S] = delta[prev] + degw[v] - 2 * cross

#     # optional endpoint pinning
#     fbit = (1 << idx[first]) if first is not None else 0
#     lbit = (1 << idx[last])  if last  is not None else 0
#     def valid(S):
#         if first is not None and not (S & fbit):
#             return False
#         if last is not None and (S & lbit) and S != full:
#             return False
#         return True

#     f = [0] * size                  # f[0] = 0 is the base case
#     parent = [-1] * size
#     for S in range(1, size):
#         if not valid(S):
#             f[S] = INF
#             continue
#         best, best_v = INF, -1
#         s = S
#         while s:
#             v = (s & -s).bit_length() - 1
#             val = f[S ^ (1 << v)]   # place v last; invalid predecessors are INF
#             if val < best:
#                 best, best_v = val, v
#             s &= s - 1
#         f[S] = delta[S] if delta[S] > best else best
#         parent[S] = best_v

#     # reconstruct the ordering (left -> right)
#     order, S = [], full
#     while S:
#         v = parent[S]
#         order.append(v)
#         S ^= 1 << v
#     order.reverse()
#     return f[full], [nodes[i] for i in order]