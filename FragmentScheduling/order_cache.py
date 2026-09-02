import numpy as np
from collections import namedtuple

from .fragments import FRAGMENTS, NotExtractable, REACH_BACK, REACH_FWD
from .primitives import build_adjacency, lr_degrees

__all__ = [
    "Proposal",
    "stamp_order",
    "order_extent",
    "occ_from_delta",
    "make_occ_cache",
    "propose_order_swap",
    "propose_order_move",
    "commit_order",
    "validate_occ_cache",
]


Proposal = namedtuple("Proposal", "order delta occ Ld Rd peak")

def stamp_order(delta, column, n_in, n_out, n_cols, sign=+1):
    # add (sign=+1) or remove (sign=-1) one spider's births and deaths
    key = (int(n_in), int(n_out))
    if key not in FRAGMENTS:
        raise NotExtractable(f"a spider types as {key} in this order; arity {sum(key)} has no  extraction fragment")
    _name, births, deaths = FRAGMENTS[key]
    for d in births:
        k = column + d + REACH_BACK
        if 0 <= k < n_cols:
            delta[k] += sign
    for d in deaths:
        k = column + d + REACH_BACK + 1
        if 0 <= k < n_cols:
            delta[k] -= sign


def order_extent(internal, Ld, Rd, node_to_idx):
    # first and last circuit column, from the fragments at the four ends  a birth sits at most 2 columns before its spider
    # and a death at most 2 after, so only positions 0 and 1 can open a column before 0 and only  n-2 and n-1 can open one past n-1
    # That makes the extent O(1) to keep up to date instead of a scan
    n = len(internal)
    c_first, c_last = 0, n - 1
    for p in (0, 1):
        if p < n:
            i = node_to_idx[internal[p]]
            _nm, births, _d = FRAGMENTS[(int(Ld[i]), int(Rd[i]))]
            for off in births:
                c_first = min(c_first, p + off)
    for p in (n - 2, n - 1):
        if p >= 0:
            i = node_to_idx[internal[p]]
            _nm, _b, deaths = FRAGMENTS[(int(Ld[i]), int(Rd[i]))]
            for off in deaths:
                c_last = max(c_last, p + off)
    return c_first, c_last


def occ_from_delta(cache, delta, internal, Ld, Rd):
    # occupancy over the circuit's real extent, boundary wires included
    # "delta" holds only the fragments own births and deaths
    # the circuits input wires are added at the first column and its output wires removed after the last
    # so neither can end before the circuit does
    c_first, c_last = order_extent(internal, Ld, Rd, cache["node_to_idx"])
    d = delta.copy()
    d[c_first + REACH_BACK] += cache["n_in_wires"]
    d[c_last + REACH_BACK + 1] -= cache["n_out_wires"]
    full = np.cumsum(d[:-1])
    lo, hi = c_first + REACH_BACK, c_last + REACH_BACK
    return full[lo:hi + 1], (c_first, c_last)


def make_occ_cache(G, order, input_node="I", output_node="O"):
    # order is the full order: input_node first, output_node last
    nodes = list(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    _adj, adj_w = build_adjacency(G)

    internal = [v for v in order if v not in (input_node, output_node)]
    n = len(internal)
    n_cols = n + REACH_BACK + REACH_FWD
    column_of = {v: p for p, v in enumerate(internal)}

    Ld, Rd = lr_degrees(order, adj_w, node_to_idx, len(nodes))

    delta = np.zeros(n_cols + 2, dtype=np.int64)
    for v in internal:
        i = node_to_idx[v]
        stamp_order(delta, column_of[v], Ld[i], Rd[i], n_cols + 1)

    cache = {"node_to_idx": node_to_idx, "adj_w": adj_w, "order": list(order),
             "internal": internal, "delta": delta, "Ld": Ld, "Rd": Rd,
             "n_cols": n_cols, "n": n, "io": (input_node, output_node),
             "n_in_wires": int(sum(adj_w.get(input_node, {}).values())),
             "n_out_wires": int(sum(adj_w.get(output_node, {}).values()))}
    cache["occ"], cache["columns"] = occ_from_delta(cache, delta, internal, Ld, Rd)
    return cache


def _restamp(cache, delta, columns, order_slice, Ld, Rd, sign):
    idx, n_cols = cache["node_to_idx"], cache["n_cols"]
    for column, v in zip(columns, order_slice):
        i = idx[v]
        stamp_order(delta, column, Ld[i], Rd[i], n_cols + 1, sign=sign)


def propose_order_swap(cache, i):
    # swap the internal vertices at positions i and i+1
    internal, idx = cache["internal"], cache["node_to_idx"]
    a, b = internal[i], internal[i + 1]
    ia, ib = idx[a], idx[b]

    delta = cache["delta"].copy()
    Ld, Rd = cache["Ld"].copy(), cache["Rd"].copy()

    _restamp(cache, delta, (i, i + 1), (a, b), Ld, Rd, -1)

    shared = cache["adj_w"].get(a, {}).get(b, 0)
    # the a-b edge changes side for both
    if shared:
        Ld[ia] += shared
        Rd[ia] -= shared
        Ld[ib] -= shared
        Rd[ib] += shared

    _restamp(cache, delta, (i, i + 1), (b, a), Ld, Rd, +1)

    new_internal = internal[:]
    new_internal[i], new_internal[i + 1] = b, a
    occ, _cols = occ_from_delta(cache, delta, new_internal, Ld, Rd)
    return Proposal(new_internal, delta, occ, Ld, Rd, int(occ.max(initial=0)))


def propose_order_move(cache, i, j):
    # move the internal vertex at position i to position j
    internal, idx, adj_w = cache["internal"], cache["node_to_idx"], cache["adj_w"]
    lo, hi = min(i, j), max(i, j)

    delta = cache["delta"].copy()
    Ld, Rd = cache["Ld"].copy(), cache["Rd"].copy()

    _restamp(cache, delta, range(lo, hi + 1), internal[lo:hi + 1], Ld, Rd, -1)

    mover = internal[i]
    im = idx[mover]
    crossed = range(i + 1, j + 1) if i < j else range(j, i)
    # moving right, the crossed vertices fall left
    sign = +1 if i < j else -1
    m_nbrs = adj_w.get(mover, {})
    for p in crossed:
        u = internal[p]
        w = m_nbrs.get(u, 0)
        if not w:
            continue
        iu = idx[u]
        Ld[im] += sign * w
        Rd[im] -= sign * w
        Ld[iu] -= sign * w
        Rd[iu] += sign * w

    new_internal = internal[:]
    new_internal.insert(j, new_internal.pop(i))
    _restamp(cache, delta, range(lo, hi + 1), new_internal[lo:hi + 1], Ld, Rd, +1)

    occ, _cols = occ_from_delta(cache, delta, new_internal, Ld, Rd)
    return Proposal(new_internal, delta, occ, Ld, Rd, int(occ.max(initial=0)))


def commit_order(cache, prop):
    # update cache if new order is commited
    cache["internal"] = prop.order
    cache["delta"], cache["occ"] = prop.delta, prop.occ
    cache["columns"] = order_extent(prop.order, prop.Ld, prop.Rd,
                                    cache["node_to_idx"])
    cache["Ld"], cache["Rd"] = prop.Ld, prop.Rd
    a, b = cache["io"]
    cache["order"] = [a] + list(prop.order) + [b]


def validate_occ_cache(cache, G):
    fresh = make_occ_cache(G, cache["order"], *cache["io"])
    for key in ("delta", "occ", "Ld", "Rd"):
        np.testing.assert_array_equal(cache[key], fresh[key], err_msg=f"cached {key!r} is stale")
