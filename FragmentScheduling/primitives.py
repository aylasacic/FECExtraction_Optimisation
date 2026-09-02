import numpy as np

__all__ = [
    "build_adjacency",
    "timesteps",
    "spider_io",
    "first_high_fan",
    "no_one_sided",
    "vtype_name",
    "spider_colour",
    "same_colour",
    "can_place",
    "lr_degrees",
    "total_energy"
]


def build_adjacency(G):
    # {v: set(nbrs)} and
    # {v: {nbr: multiplicity}}
    adj = {v: set() for v in G.nodes()}
    adj_w = {v: {} for v in G.nodes()}
    try:
        edges = [(u, w) for u, w, _ in G.edges(keys=True)]
    except TypeError:
        edges = list(G.edges())
    for u, w in edges:
        # skip loops (shoudl be removed by cleanup)
        if u == w:
            continue
        adj[u].add(w)
        adj[w].add(u)
        adj_w[u][w] = adj_w[u].get(w, 0) + 1
        adj_w[w][u] = adj_w[w].get(u, 0) + 1
    return adj, adj_w


def timesteps(sched):
    ts = {v: t for t, layer in enumerate(sched) for v in layer}
    ts["I"] = -1
    ts["O"] = len(sched)
    return ts


def spider_io(sched, adj_w, node_to_ts=None):
    # {node: (n_in, n_out)} for every scheduled node
    ts = node_to_ts if node_to_ts is not None else timesteps(sched)
    io = {}
    for t, layer in enumerate(sched):
        for v in layer:
            n_in = n_out = 0
            for nbr, w in adj_w.get(v, {}).items():
                tn = ts.get(nbr)
                if tn is None:
                    continue
                if tn < t:
                    n_in += w
                elif tn > t:
                    n_out += w
            io[v] = (n_in, n_out)
    return io


def first_high_fan(sched, adj_w, max_fan = 2, io = None):
    # first one-sided spider fanning past max_fan: (v, t, kind) or None
    io = io if io is not None else spider_io(sched, adj_w)
    for t, layer in enumerate(sched):
        for v in layer:
            n_in, n_out = io[v]
            if n_in == 0 and n_out > max_fan:
                return v, t, "split"
            if n_out == 0 and n_in > max_fan:
                return v, t, "merge"
    return None


def no_one_sided(adj_w, sched, max_fan = 2, io = None):
    return first_high_fan(sched, adj_w, max_fan, io) is None


def vtype_name(raw):
    # "Z", "X" or None, for a raw vtype attribute that may be a pyzx
    # VertexType enum, a plain string, or missing
    if raw is None:
        return None
    name = getattr(raw, "name", str(raw)).upper()
    if name == "Z" or name.endswith(".Z"):
        return "Z"
    if name == "X" or name.endswith(".X"):
        return "X"
    return None


def spider_colour(G, v):
    # Z or X OR None for boundaries/helpers so they are never paired
    if v not in G.nodes:
        return None
    return vtype_name(G.nodes[v].get("vtype"))


def same_colour(G, u, v):
    # return if two spiders under observation are of the same colour
    cu, cv = spider_colour(G, u), spider_colour(G, v)
    return cu is not None and cu == cv


def can_place(node, layer, adj, G, allow_co_measure = False, layer_idx = None,
              n_layers = None, allow_last_layer_measure = True):
    # may node join layer?

    # mo in-layer neighbour -> yes, ordinary parallel node
    # co-measure off -> no, any adjacency is a real cut edge
    # co-measure on, all in-layer
    # TO BE FIXED
    # neighbours share node's colour -> yes, it joins/extends/fuses one multi-body XX/ZZ parity measurement

    # any differently coloured  in-layer neighbour -> no.

    # with allow_last_layer_measure = False, a co-measure additionally needs more
    # than one layer left before output, because its ancilla lives on boundaries
    # t-1...t+2 and would otherwise run past the output boundary (the t+2 part)

    nbrs = [w for w in layer if w in adj[node]]
    if not nbrs:
        return True
    if not allow_co_measure:
        return False
    if not all(same_colour(G, node, w) for w in nbrs):
        return False
    if (not allow_last_layer_measure) and layer_idx is not None and n_layers is not None:
        if n_layers - layer_idx <= 1:
            return False
    return True


def lr_degrees(order, adj_w, node_to_idx, n_nodes):
    # edge weight to the left\right of each vertex in order
    Ld = np.zeros(n_nodes, dtype=np.int64)
    Rd = np.zeros(n_nodes, dtype=np.int64)
    position = {v: p for p, v in enumerate(order)}

    for p, v in enumerate(order):
        i = node_to_idx[v]
        for nbr, w in adj_w.get(v, {}).items():
            q = position.get(nbr)
            if q is None:
                continue
            if q < p:
                Ld[i] += w
            elif q > p:
                Rd[i] += w
    return Ld, Rd


def total_energy(cut_arr, qubit_limit, lam = 1000, p = 8):
    # smooth max of the cut profile plus a quadratic over-limit penalty
    arr = np.asarray(cut_arr, dtype = np.float64)
    if arr.size == 0:
        return 0.0

    maximum = float(arr.max())
    if maximum == 0:
        smooth_maximum = 0.0
    else:
        smooth_maximum = maximum * (np.mean((arr / maximum) ** p)) ** (1.0 / p)

    excess = np.maximum(0.0, arr - qubit_limit)
    return float(smooth_maximum + lam * np.sum(excess ** 2))
