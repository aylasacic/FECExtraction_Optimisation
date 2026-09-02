from .layers import leg_directions_ok, trial_move
from .occupancy import occupancy_profile
from .primitives import can_place, no_one_sided

__all__ = ["try_move", "refine", "move_earliest"]


def try_move(G, sched, adj_w, adj, node, src, tgt, limit, *, allow_co_measure, allow_last_layer_measure,
             check_place = True, pure_parallel = False, check_one_sided = False, max_fan = 2):
    # one trial move plus every legality check
    # returns (trial, Profile) or None
    L = len(sched)
    if tgt < L:
        if pure_parallel:
            if any(x in adj[node] for x in sched[tgt]):
                return None
        elif check_place and not can_place(
                node, sched[tgt], adj, G, allow_co_measure, layer_idx = tgt,
                n_layers = L, allow_last_layer_measure = allow_last_layer_measure):
            return None

    trial = trial_move(sched, node, src, tgt)
    if not leg_directions_ok(trial):
        return None
    if check_one_sided and not no_one_sided(adj_w, trial, max_fan):
        return None
    return trial, occupancy_profile(G, trial, adj_w, limit)


def refine(G, schedule, adj_w, adj, limit, *, mode, max_passes = 30, max_fan = 2, allow_co_measure = False,
           allow_last_layer_measure = True, require_no_one_sided = None, protect_peak = False, window = None, 
           max_iters = 10000):

    # local search over single-node layer moves.
    # mode = "depth" -> pull nodes into earlier layers
    # mode = "volume" -> minimise sum(aug) at fixed depth
    # mode = "width" -> escape an infeasible peak
    sched = [s[:] for s in schedule if s]

    if mode == "width":
        return _refine_width(G, sched, adj_w, adj, limit, max_fan = max_fan, allow_co_measure = allow_co_measure,
                             allow_last_layer_measure = allow_last_layer_measure, window = window, max_iters = max_iters)

    if require_no_one_sided is None:
        require_no_one_sided = (mode == "volume")

    if mode == "volume":
        prof0 = occupancy_profile(G, sched, adj_w)
        current_volume = prof0.volume
        # protect_peak keeps the width already paid for
        # otherwise the polish may spend the whole budget for a deeper volume cut
        cap = prof0.peak if protect_peak else limit
    else:
        current_volume, cap = None, limit

    common = dict(allow_co_measure = allow_co_measure, allow_last_layer_measure = allow_last_layer_measure,
                  check_one_sided = require_no_one_sided, max_fan = max_fan)

    for _ in range(max_passes):
        improved = False
        for src in range(len(sched)):
            for node in list(sched[src]):
                # depth: earlier layers only, so depth can only shrink
                # volume: existing layers only, so depth can never grow.
                targets = range(src) if mode == "depth" else range(len(sched))
                choice = None
                for tgt in targets:
                    if tgt == src:
                        continue
                    res = try_move(G, sched, adj_w, adj, node, src, tgt, cap, **common)
                    if res is None or not res[1].feasible:
                        continue
                    if mode == "depth":
                        # first improvement wins
                        choice = tgt
                        break
                    # best strict improvement wins
                    v = res[1].volume
                    if v < current_volume and (choice is None or v < choice[0]):
                        choice = (v, tgt)
                if choice is None:
                    continue
                if mode == "volume":
                    current_volume, tgt = choice
                else:
                    tgt = choice
                sched[src].remove(node)
                sched[tgt].append(node)
                improved = True
        sched = [s for s in sched if s]
        if not improved:
            break
    return sched


def _refine_width(G, sched, adj_w, adj, limit, *, max_fan, allow_co_measure, allow_last_layer_measure, window, max_iters):
    # push nodes off the most overloaded boundary until the schedule fits
    prof = occupancy_profile(G, sched, adj_w, limit)

    for _ in range(max_iters):
        if prof.feasible:
            break
        L = len(sched)
        # aug[0] is the input boundary and aug[k+1] is the gap AFTER layer k, so the nodes touching boundary k are the
        # ones in layers k-1 and k
        k = max(range(prof.depth), key=lambda i: prof.occ[i])
        centre = min(max(k - 1, 0), L - 1)
        cands = list(sched[centre])
        if centre + 1 < L:
            cands += list(sched[centre + 1])

        if window is None:
            targets = list(range(L)) + [L]
        else:
            targets = list(range(max(0, centre - window), min(L, centre + 2 + window))) + [L]

        node_layer = {v: i for i, s in enumerate(sched) for v in s}
        best = None
        for node in cands:
            src = node_layer[node]
            for tgt in targets:
                if tgt == src:
                    continue
                res = try_move(G, sched, adj_w, adj, node, src, tgt, limit, allow_co_measure = allow_co_measure,
                               allow_last_layer_measure = allow_last_layer_measure)
                if res is None:
                    continue
                if best is None or res[1].peak < best[0]:
                    best = (res[1].peak, res[0], res[1])

        if best is None or best[0] >= prof.peak:
            break
        _, sched, prof = best

    return sched, prof.feasible, prof.peak


def move_earliest(G, sched, adj_w, adj, mover, src, limit, *, max_fan = 2, allow_co_measure = False, allow_last_layer_measure = True,
                  require_no_one_sided = True, pure_parallel = False):
    # earliest existing layer that accepts mover, or None
    # earliest-first keeps co-measures as early as possible
    # pure_parallel = True only lands where mover has no neighbour at all (forms/joins no group)
    for tgt in range(len(sched)):
        if tgt == src:
            continue
        res = try_move(G, sched, adj_w, adj, mover, src, tgt, limit, allow_co_measure = allow_co_measure,
                       allow_last_layer_measure = allow_last_layer_measure, pure_parallel = pure_parallel,
                       check_one_sided = require_no_one_sided, max_fan = max_fan)
        if res is not None and res[1].feasible:
            return [s for s in res[0] if s]
    return None
