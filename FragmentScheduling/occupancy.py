from collections import Counter, defaultdict

from .fragments import FRAGMENTS, NotExtractable, Occupancy, spider_typing
from .primitives import spider_io, timesteps

__all__ = [
    "check_no_self_loops",
    "check_schedule_covers_graph",
    "check_no_same_layer_edges",
    "occupancy_profile",
    "explain_column",
]


def check_no_self_loops(G):
    # build_adjacency drops self-loops, so a spider carrying one would be
    # typed at a lower arity than it really has and get the wrong fragment
    # clean-up removes them -> this makes sure of it rather than mistyping
    loops = [u for u, v, _k in G.edges(keys = True) if u == v]
    if loops:
        raise NotExtractable(f"self-loops on {sorted(set(map(repr, loops)))[:5]};clean-up first.")


def check_schedule_covers_graph(G, sched, input_node="I", output_node="O"):
    # every interior spider scheduled exactly once and nothing else
    # a spider left out of the schedule still has its boundary edges counted in n_in_wires / n_out_wires,
    # but never gets a typing, so nothing ever consumes the line it feeds (+1 on every line)
    scheduled = [v for layer in sched for v in layer]
    repeats = [v for v, c in Counter(scheduled).items() if c > 1]
    if repeats:
        raise NotExtractable(f"scheduled more than once: {repeats[:5]}")

    interior = set(G.nodes()) - {input_node, output_node}
    missing = interior - set(scheduled)
    extra = set(scheduled) - interior
    if missing or extra:
        raise NotExtractable(f"schedule does not match the graph: {len(missing)} spider(s) in the ")


def check_no_same_layer_edges(G, sched, adj_w):
    # a spider ordering needs pi(u) != pi(v) for every adjacent pair
    # two adjacent spiders in one layer leave the edge between them with no time
    # direction, so neither spider has a typing and neither has a fragment
    for t, layer in enumerate(sched):
        seen = set(layer)
        for u in layer:
            for v in adj_w.get(u, {}):
                if v in seen and v != u:
                    raise NotExtractable(f"spiders {u!r} and {v!r} are adjacent and both in layer")


def occupancy_profile(G, schedule, adj_w, limit=None, *, input_node="I", output_node="O"):
    # qubits in use at every column of the extracted circuit
    # "occ" is indexed by circuit column, "columns" gives the first and last column it covers,
    # "peak" is the qubit count and "depth" the column count
    # the circuit's own input and output wires span the WHOLE extent: the inputs are live from the first column,
    # before any spider acts and the outputs stay live to the last column AND Neither can end early
    # a CAT state in the first layer opens columns before layer 0 and the inputs are already there
    # and a merge or CAT adjoint in the last layer opens columns past layer L-1 with the outputs still running through them
    sched = [s for s in schedule if s]
    if not sched:
        return Occupancy([], 0, 0, 0, (0, 0), limit is None or 0 <= limit, {})

    # quality control
    check_no_self_loops(G)
    check_schedule_covers_graph(G, sched, input_node, output_node)
    check_no_same_layer_edges(G, sched, adj_w)

    node_to_ts = timesteps(sched)
    io = spider_io(sched, adj_w, node_to_ts)
    L = len(sched)

    births, deaths, typings = [], [], {}
    for t, layer in enumerate(sched):
        for v in layer:
            key = spider_typing(io, v)
            typings[v] = key
            _name, b, d = FRAGMENTS[key]
            births.extend(t + x for x in b)
            deaths.extend(t + x for x in d)

    # the circuit runs from the earliest column any fragment reaches back to, to the latest column any fragment reaches forward to
    c_first = min([0] + births)
    c_last = max([L - 1] + deaths)

    n_in_wires = int(sum(adj_w.get(input_node, {}).values()))
    n_out_wires = int(sum(adj_w.get(output_node, {}).values()))
    births += [c_first] * n_in_wires
    deaths += [c_last] * n_out_wires

    width = c_last - c_first + 1
    delta = [0] * (width + 1)
    for c in births:
        delta[c - c_first] += 1
    # the line is live THROUGH column c
    for c in deaths:
        delta[c - c_first + 1] -= 1

    occ, running = [], 0
    for i in range(width):
        running += delta[i]
        occ.append(running)

    if min(occ) < 0:
        raise NotExtractable("negative occupancy: a line dies before it is born (schedule inconsistent)")
    leftover = running + delta[width]
    if leftover != 0:
        raise NotExtractable(f"{leftover} qubit line(s) never end: every line has to be consumed")

    peak = max(occ, default=0)
    return Occupancy(occ=occ, peak=peak, volume=sum(occ), depth=width,
                     columns=(c_first, c_last), feasible=(limit is None or peak <= limit), typings=typings)


# ---------------------------------------------------------------------------
# DEBUGGING
# ---------------------------------------------------------------------------
def explain_column(G, schedule, adj_w, column, input_node="I", output_node="O"):
    sched = [s for s in schedule if s]
    L = len(sched)
    ts = timesteps(sched)
    io = spider_io(sched, adj_w, ts)

    births_all, deaths_all = [], []
    for t, layer in enumerate(sched):
        for v in layer:
            _n, b, d = FRAGMENTS[spider_typing(io, v)]
            births_all += [t + x for x in b]
            deaths_all += [t + x for x in d]
    c_first = min([0] + births_all)
    c_last = max([L - 1] + deaths_all)

    oriented = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        if u == v:
            continue
        tu, tv = ts.get(u), ts.get(v)
        if tu is None or tv is None or tu == tv:
            continue
        a, b = (u, v) if tu < tv else (v, u)
        oriented[a].append((b, k, "out"))
        oriented[b].append((a, k, "in"))

    def eid(a, b, k):
        return (a, b, k) if ts[a] <= ts[b] else (b, a, k)

    lines, edge_line = [], {}
    for nbr, k, _d in oriented.get(input_node, []):
        edge_line[eid(input_node, nbr, k)] = len(lines)
        lines.append({"born": c_first, "by": f"input wire -> {nbr!r}", "dies": None, "at": None})

    for t, layer in enumerate(sched):
        for v in layer:
            key = spider_typing(io, v)
            name, births, deaths = FRAGMENTS[key]
            ins = [eid(n, v, k) for n, k, d in oriented[v] if d == "in"]
            outs = [eid(v, n, k) for n, k, d in oriented[v] if d == "out"]
            arriving = [edge_line[e] for e in ins]
            through = min(key)
            for e, ln in zip(outs[:through], arriving[:through]):
                edge_line[e] = ln
            for e, off in zip(outs[through:], births):
                edge_line[e] = len(lines)
                lines.append({"born": t + off, "by": f"{v!r} ({name}) t{off:+d}", "dies": None, "at": None})
            for ln, off in zip(arriving[through:], deaths):
                lines[ln]["dies"] = t + off
                lines[ln]["at"] = f"{v!r} ({name}) t{off:+d}"

    for nbr, k, _d in oriented.get(output_node, []):
        ln = edge_line[eid(nbr, output_node, k)]
        lines[ln]["dies"] = c_last
        lines[ln]["at"] = f"output wire from {nbr!r}"

    live = [l for l in lines if l["born"] <= column <= l["dies"]]
    print(f"column {column} of {c_first}-{c_last}: {len(live)} qubits")
    for l in sorted(live, key=lambda x: x["born"]):
        print(f"born c{l['born']:<4} by {l['by']:<34} | dies c{l['dies']:<4} at {l['at']}")
    return live
