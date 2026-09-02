from .co_measure import co_measure_cleanup
from .primitives import build_adjacency, first_high_fan, no_one_sided, spider_io, timesteps
from .refine import refine

__all__ = ["relink_adjacency", "unfuse", "optimize_depth_with_unfusion"]


def relink_adjacency(adj, adj_w, v, w, moved):
    # patch adjacency after unfusing w off v and moving moved legs to it
    # moved lists one neighbour per moved edge (repeats mean parallel edges)
    adj.setdefault(w, set())
    adj_w.setdefault(w, {})

    for nbr in moved:
        adj_w[v][nbr] -= 1
        adj_w[nbr][v] -= 1
        # last parallel edge v-nbr gone
        if adj_w[v][nbr] == 0:
            del adj_w[v][nbr]
            del adj_w[nbr][v]
            adj[v].discard(nbr)
            adj[nbr].discard(v)
        adj_w[w][nbr] = adj_w[w].get(nbr, 0) + 1
        adj_w[nbr][w] = adj_w[nbr].get(w, 0) + 1
        adj[w].add(nbr)
        adj[nbr].add(w)

    # the fusion wire
    adj_w[v][w] = adj_w[v].get(w, 0) + 1
    adj_w[w][v] = adj_w[w].get(v, 0) + 1
    adj[v].add(w)
    adj[w].add(v)
    return adj, adj_w


def unfuse(G, schedule, colour_map = None, pos = None, max_fan = 2, link_attrs = None):
    # split one-sided spiders that fan out past max_fan into two spiders.
    # each pass finds the first offender, peels its heavy-side legs onto a fresh same-colour phase-0
    # spider and schedules that spider in the gap between the two halves (opening a layer only if there is no gap)
    G = G.copy()
    sched = [list(s) for s in schedule if s]
    pos = dict(pos) if pos is not None else None
    # fusion wire is a plain edge
    link_attrs = dict(link_attrs or {})
    adj, adj_w = build_adjacency(G)

    def new_id():
        return max((n for n in G.nodes if isinstance(n, int)), default = -1) + 1

    for _ in range(50 * G.number_of_nodes() + 100):
        node_to_ts = timesteps(sched)
        target = first_high_fan(sched, adj_w, max_fan, io = spider_io(sched, adj_w, node_to_ts))
        if target is None:
            break
        v, t, kind = target

        # heavy-side legs, keeping multigraph keys and edge attrs
        legs = []
        for _, nbr, k, data in G.edges(v, keys = True, data = True):
            tn = node_to_ts.get(nbr)
            if tn is None:
                continue
            if (kind == "split" and tn > t) or (kind == "merge" and tn < t):
                legs.append((tn, nbr, k, dict(data)))

        if kind == "split":
            legs.sort(key=lambda e: e[0])      # keep the earliest out-leg on v
            move = legs[1:]
            gap_lo, gap_hi = t, min(e[0] for e in move)
        else:
            legs.sort(key=lambda e: e[0], reverse=True)   # keep the latest in-leg
            move = legs[1:]
            gap_lo, gap_hi = max(e[0] for e in move), t

        # fresh spider: v's attrs and colour, phase 0 (the phase stays on v)
        w = new_id()
        attrs = dict(G.nodes[v])
        attrs["key"] = f"U_{w}"
        if "phase" in attrs:
            attrs["phase"] = 0
        G.add_node(w, **attrs)
        if colour_map is not None and v in colour_map:
            colour_map[w] = colour_map[v]
        if pos is not None and v in pos:
            x, y = pos[v]
            pos[w] = (x + 0.4, y + 0.25)

        # edge attrs survive the reroute
        for _, nbr, k, data in move:
            G.remove_edge(v, nbr, k)
            G.add_edge(w, nbr, **data)
        G.add_edge(v, w, **link_attrs)
        relink_adjacency(adj, adj_w, v, w, [nbr for _, nbr, _, _ in move])

        # w's only neighbours are v and the moved legs, so any layer strictly
        # inside (gap_lo, gap_hi) is adjacency-free by construction
        if gap_hi - gap_lo > 1:
            sched[gap_lo + 1].append(w)
        else:
            sched.insert(gap_lo + 1, [w])
    else:
        print("unfuse did not converge")
        return None

    return G, sched, adj, adj_w, colour_map, pos


def optimize_depth_with_unfusion(G, schedule, limit, colour_map = None, pos = None, max_rounds = 8, compact_passes = 30,
                                 max_fan = 2, link_attrs = None, allow_co_measure = False, polish_volume = False,
                                 protect_peak = True, allow_last_layer_measure = True):
    G = G.copy()
    sched = [list(s) for s in schedule if s]
    adj = adj_w = None
    common = dict(allow_co_measure = allow_co_measure, allow_last_layer_measure = allow_last_layer_measure)

    for _ in range(max_rounds):
        out = unfuse(G, sched, colour_map = colour_map, pos = pos, max_fan = max_fan, link_attrs = link_attrs)
        if out is None:
            break
        G, sched, adj, adj_w, colour_map, pos = out
        depth_before = len(sched)
        sched = refine(G, sched, adj_w, adj, limit, mode = "depth", max_passes = compact_passes, max_fan = max_fan, **common)
        if len(sched) == depth_before and no_one_sided(adj_w, sched, max_fan):
            break

    if adj_w is None:
        return G, sched, adj, adj_w, colour_map, pos

    if polish_volume:
        sched = refine(G, sched, adj_w, adj, limit, mode = "volume", max_passes = compact_passes, max_fan = max_fan,
                       protect_peak = protect_peak, **common)
    sched = co_measure_cleanup(G, sched, adj_w, adj, limit, max_fan = max_fan, **common)
    return G, sched, adj, adj_w, colour_map, pos
