from collections import defaultdict
from .primitives import can_place, same_colour

__all__ = [
    "vert_info",
    "leg_directions_ok",
    "greedy_layers",
    "components",
    "joint_groups_in_layer"
]


def vert_info(node):
    # ('parent', 'I'|'O') for a split leg node, else None
    if isinstance(node, tuple) and len(node) >= 3 and node[0] == 'leg' and node[2] in ('I', 'O'):
        return node[1], node[2]
    return None


def leg_directions_ok(schedule):
    # I legs must sit before their parent
    # O legs after
    layer = {v: t for t, s in enumerate(schedule) for v in s}
    for v, t in layer.items():
        info = vert_info(v)
        if not info:
            continue
        par = layer.get(info[0])
        if par is None:
            continue
        if (info[1] == 'I' and t >= par) or (info[1] == 'O' and t <= par):
            return False
    return True


def trial_move(schedule, node, src, tgt):
    # copy of schedule with node moved from layer src to layer tgt
    new = [s[:] for s in schedule]
    new[src].remove(node)
    if tgt >= len(new):
        new.append([node])
    else:
        new[tgt].append(node)
    return [s for s in new if s]


# kept under the notebook's private name too, so existing call sites work
_trial_move = trial_move


def greedy_layers(G, order, adj, allow_co_measure=False, allow_last_layer_measure=True):
    # pack "order" into as few layers as adjacency allows, honouring legs
    # input legs are forced before their parent and O legs after, so the output can
    # never be scheduled ahead of an internal node
    order = [v for v in order if v not in ("I", "O")]
    order_set = set(order)

    preds = defaultdict(set)
    for v in order:
        info = vert_info(v)
        if not info:
            continue
        parent, side = info
        if parent not in order_set:
            continue
        if side == 'I':
            # leg before parent
            preds[parent].add(v)
        else:
            # parent before leg
            preds[v].add(parent)

    layers, layer_sets, layer_of = [], [], {}
    # SA order is the tie-breaker
    remaining = list(order)
    while remaining:
        progressed, leftover = False, []
        for v in remaining:
            if not all(p in layer_of for p in preds[v]):
                leftover.append(v)
                continue
            j = max((layer_of[p] + 1 for p in preds[v]), default=0)

            while j < len(layers) and not can_place(
                    v, layer_sets[j], adj, G, allow_co_measure, layer_idx=j,
                    n_layers=len(layers),
                    allow_last_layer_measure=allow_last_layer_measure):
                j += 1
            if j == len(layers):
                layers.append([])
                layer_sets.append(set())
            layers[j].append(v)
            layer_sets[j].add(v)
            layer_of[v] = j
            progressed = True
        remaining = leftover
        # cyclic leftover: give each its own layer
        if not progressed:
            layers.extend([v] for v in remaining)
            break
    return [s for s in layers if s]


# ---------------------------------------------------------------------------
# CONNECTED COMPONENTS [TBD -> not yet used]
# ---------------------------------------------------------------------------
def components(nodes, edges):
    nodes = list(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in edges:
        iu, iv = index.get(u), index.get(v)
        if iu is None or iv is None:
            continue
        ru, rv = find(iu), find(iv)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)

    groups = defaultdict(list)
    for i, v in enumerate(nodes):
        groups[find(i)].append(v)
    return [tuple(groups[r]) for r in sorted(groups)]


def joint_groups_in_layer(layer, adj_w, G):
    # co-measured groups inside one layer: 
    # maximal sets of same-coloured spiders that are adjacent within the layer
    # singletons are not groups -> one spider on its own is an ordinary parallel node, not a joint read-out
    layer = list(layer)
    in_layer = set(layer)
    edges = []
    for u in layer:
        for v in adj_w.get(u, {}):
            if v != u and v in in_layer and same_colour(G, u, v):
                edges.append((u, v))
    return [c for c in components(layer, edges) if len(c) > 1]
