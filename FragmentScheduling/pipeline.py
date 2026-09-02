from collections import namedtuple

from .co_measure import co_measure_cleanup
from .layers import greedy_layers
from .occupancy import occupancy_profile
from .refine import refine
from .unfusion import optimize_depth_with_unfusion

__all__ = ["Sched", "as_sched", "schedule_min_depth"]


Sched = namedtuple("Sched", "G schedule depth peak feasible colour_map pos adj adj_w")


def as_sched(G, schedule, limit, *, colour_map = None, pos = None, adj = None, adj_w = None, allow_co_measure = False):
    prof = occupancy_profile(G, schedule, adj_w, limit)
    return Sched(G = G, schedule = schedule, depth = len([s for s in schedule if s]), peak = prof.peak,
                 feasible = prof.feasible, colour_map = colour_map, pos = pos, adj = adj, adj_w = adj_w)


def schedule_min_depth(G, order, limit, adj, adj_w, *, repair_window = None, compact_passes = 30,
                       try_repair = True, allow_co_measure = False, allow_last_layer_measure = True, use_unfusion = False,
                       colour_map = None, pos = None, max_fan = 2, link_attrs = None, polish_volume = False):
    # shallowest schedule for `order` that still fits `limit`.

    # 1. greedy layering
    # 2. serial fallback, one node per layer (safest but deepest)
    # 3. width repair if even that overflows
    # 4. depth compaction
    # 5. optional unfusion + volume polish, then co-measure cleanup

    # eeturns a Sched, or None if nothing feasible could be built

    common = dict(allow_co_measure = allow_co_measure, allow_last_layer_measure = allow_last_layer_measure)

    layers = greedy_layers(G, order, adj=adj, **common)
    if not occupancy_profile(G, layers, adj_w, limit).feasible:
        layers = [[v] for v in order if v not in ("I", "O")]
        if not occupancy_profile(G, layers, adj_w, limit).feasible:
            if not try_repair:
                return None
            layers, ok, _ = refine(G, layers, adj_w, adj, limit, mode = "width",
                                   window = repair_window, max_fan = max_fan, **common)
            if not ok:
                return None
        layers = refine(G, layers, adj_w, adj, limit, mode = "depth", max_passes = compact_passes, max_fan = max_fan, **common)

    res = as_sched(G, layers, limit, colour_map = colour_map, pos = pos, adj = adj, adj_w = adj_w)
    if not res.feasible:
        return res

    if use_unfusion:
        # unfusion changes the depth again and runs its own cleanup, so there is
        # no point cleaning the pre-unfusion layers first
        G2, sched2, adj2, adj_w2, cmap2, pos2 = optimize_depth_with_unfusion(G, layers, limit, colour_map = colour_map,
                                                                            pos = pos, max_fan = max_fan,
                                                                            link_attrs = link_attrs, compact_passes = compact_passes,
                                                                            polish_volume = polish_volume, **common)
        return as_sched(G2, sched2, limit, colour_map = cmap2, pos = pos2, adj = adj2, adj_w = adj_w2)

    layers = co_measure_cleanup(G, layers, adj_w, adj, limit, max_fan = max_fan, **common)
    return as_sched(G, layers, limit, colour_map = colour_map, pos = pos, adj = adj, adj_w = adj_w)
