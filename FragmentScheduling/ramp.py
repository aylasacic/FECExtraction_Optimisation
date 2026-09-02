import time
from collections import Counter

from Utils.print_OLA_graphs import plot

from .fragments import FRAGMENTS, NotExtractable
from .layers import greedy_layers
from .occupancy import occupancy_profile
from .pipeline import schedule_min_depth
from .primitives import build_adjacency
from .schedulers import sched_anneal, sched_cap_sweep, sched_serial

__all__ = ["progressive_depth_search", "compare_ramps"]


def _ramp_scheduler(name):
    def plain(fn):
        def run(ctx, o, q, start = None, **kw):
            return fn(ctx, o, q, start, **kw), ctx
        return run

    if name == "serial":
        return plain(lambda c, o, q, s, **k: [[v] for v in o])
    if name == "greedy":
        return plain(lambda c, o, q, s, **k: greedy_layers(
            c["G"], o, adj=c["adj"],
            allow_last_layer_measure=k.get("allow_last_layer_measure", True)))
    if name == "cap sweep":
        return plain(lambda c, o, q, s, **k: sched_cap_sweep(
            c["G"], o, c["adj"], c["adj_w"], q))

    if name == "pipeline":
        def run(ctx, o, q, start = None, use_unfusion = False, polish_volume = False,
                allow_last_layer_measure = True, **kw):
            res = schedule_min_depth(
                ctx["G"], o, q, ctx["adj"], ctx["adj_w"],
                colour_map=ctx["colour_map"], pos=ctx["pos"],
                use_unfusion=use_unfusion, polish_volume=polish_volume,
                allow_last_layer_measure=allow_last_layer_measure, **kw)
            if res is None or not res.feasible:
                return None, ctx
            return res.schedule, {"G": res.G, "adj": res.adj, "adj_w": res.adj_w, "colour_map": res.colour_map, "pos": res.pos}
        return run

    if name == "anneal":
        def run(ctx, o, q, start = None, budget = 40000, seed = 0, use_unfusion = False,
                polish_volume = False, allow_last_layer_measure = True, **kw):
            if start is None:
                start, ctx = _ramp_scheduler("pipeline")(
                    ctx, o, q, None, use_unfusion=use_unfusion,
                    polish_volume=polish_volume,
                    allow_last_layer_measure = allow_last_layer_measure, **kw)
            out = sched_anneal(ctx["G"], o, ctx["adj"], ctx["adj_w"], q,
                               budget = budget, seed = seed,
                               start = [list(s) for s in start] if start else None)
            return (out if out is not None else start), ctx
        return run

    raise ValueError(f"unknown scheduler {name!r}")


def progressive_depth_search(G, sa_fn, q_min, q_max, *, scheduler = "anneal",
                             step = 1, order = None, colour_map = None, pos = None,
                             sa_kwargs = None, sched_kwargs = None, seeds = (23,),
                             anneal_budget = 40000, anneal_seed = 0,
                             target_depth = None, patience = None,
                             allow_co_measure = False, use_unfusion = False,
                             polish_volume = False, allow_last_layer_measure = True,
                             verbose = True, plot_each = True):
    if allow_co_measure:
        raise NotExtractable(
            "allow_co_measure=True puts adjacent spiders in one layer, which "
            "leaves the edge between them with no time direction. Neither "
            "spider then has a typing at the moment [TBD]")

    adj, adj_w = build_adjacency(G)
    sa_kwargs = dict(sa_kwargs or {})
    sched_kwargs = dict(sched_kwargs or {})

    # 1. the order, annealed once and then left alone
    sa_width = None
    if order is None:
        for sd in seeds:
            kw = dict(sa_kwargs, seed = sd, qubit_limit = q_max, stop_when_feasible = False)
            cand_order, _e, cand_width, _h, _b = sa_fn(G, **kw)
            if sa_width is None or cand_width < sa_width:
                order, sa_width = cand_order, cand_width
    order = [v for v in order if v not in ("I", "O")]
    q_min = sa_width if q_min is None else q_min

    # 2. the two limits worth knowing before the ramp starts
    serial = [[v] for v in order]

    serial_prof = occupancy_profile(G, serial, adj_w)

    if verbose:
        print(serial_prof.occ)

    floor = serial_prof.peak
    base_prof = occupancy_profile(G, greedy_layers(G, order, adj = adj), adj_w)
    if verbose:
        print(f"[{scheduler}] order needs {floor} qubits serially over "
              f"{serial_prof.depth} columns -> fully parallel is {base_prof.depth} columns at {base_prof.peak} qubits")

    run = _ramp_scheduler(scheduler)
    extra = dict(sched_kwargs, use_unfusion = use_unfusion, 
                    polish_volume = polish_volume, allow_last_layer_measure = allow_last_layer_measure)
    if scheduler == "anneal":
        extra.update(budget = anneal_budget, seed = anneal_seed)

    ctx0 = {"G": G, "adj": adj, "adj_w": adj_w, "colour_map": colour_map,
            "pos": pos}
    results, best, stale = [], None, 0
    carry, carry_ctx = None, None
    prev_plotted = None

    for q in range(max(q_min, floor), q_max + 1, step):
        t0 = time.time()
        # only offer the carried schedule when it belongs to T
        # THIS graph unfusion replaces G, 
        # and a schedule for the old one is meaningless
        seed_sched = carry if (carry_ctx is not None and carry_ctx["G"] is ctx0["G"]) else None
        try:
            sched, ctx = run(ctx0, order, q, start = seed_sched, **extra)
        except NotExtractable:
            sched, ctx = None, ctx0
        prof = occupancy_profile(ctx["G"], sched, ctx["adj_w"], q) if sched else None
 
        if verbose and prof is not None:
            print(prof.occ)

        # 3. keep whichever is better: this budget's answer, or the one carried
        #    up from the budget below, which is still feasible now
        if carry is not None and carry_ctx["G"] is ctx["G"]:
            cprof = occupancy_profile(ctx["G"], carry, ctx["adj_w"], q)
            if prof is None or not prof.feasible or (
                    cprof.feasible
                    and (cprof.depth, cprof.volume) < (prof.depth, prof.volume)):
                sched, prof = carry, cprof

        secs = time.time() - t0
        if prof is None or not prof.feasible:
            results.append({"q": q, "scheduler": scheduler, "feasible": False,
                            "depth": None, "peak": prof.peak if prof else None,
                            "seconds": secs})
            if verbose:
                print(f"  q={q:>3}: does not fit"
                      + (f" (needs {prof.peak})" if prof else ""))
            continue

        carry, carry_ctx = [list(s) for s in sched], ctx
        rec = {"q": q, "scheduler": scheduler, "feasible": True,
                # circuit columns
               "depth": prof.depth,               
               "layers": len([s for s in sched if s]),
               # qubits actually used
               "peak": prof.peak,                 
               "volume": prof.volume,
               "reserved": q * prof.depth,
               "idle": q * prof.depth - prof.volume,
               "occupancy": list(prof.occ), "columns": prof.columns,
               "fragments": dict(Counter(FRAGMENTS[k][0]
                                         for k in prof.typings.values())),
               "schedule": [list(s) for s in sched], "graph": ctx["G"],
               "adj_w": ctx["adj_w"], "colour_map": ctx["colour_map"],
               "pos": ctx["pos"], "seconds": secs}
        results.append(rec)
        if verbose:
            print(f"q={q:>3}: {prof.depth:>4} columns, {prof.peak:>3} qubits, "
                  f"volume {prof.volume:>5}, idle {rec['idle']:>5}, "
                  f"{secs:>5.1f}s")

        if plot_each and sched != prev_plotted:
            plot(ctx["G"], order = sched, cutwidths = prof.peak, depth = prof.depth,
                 title_prefix = (f"{scheduler} | {prof.peak} qubits, {prof.depth} columns, volume {prof.volume}, q = {q}"), 
                 size = 10, draw_labels = True, colour_in = True)
            prev_plotted = [list(s) for s in sched]

        if best is None or prof.depth < best["depth"]:
            best, stale = rec, 0
        else:
            stale += 1
        if target_depth is not None and prof.depth <= target_depth:
            break
        if patience is not None and stale >= patience:
            if verbose:
                print(f"  depth flat for {patience} budgets, stopping")
            break

    return {"order": order, "scheduler": scheduler, "ramp": results,
            "best": best, "sa_width": sa_width, "floor": floor,
            "parallel_limit": {"depth": base_prof.depth,
                               "peak": base_prof.peak}}


def compare_ramps(G, sa_fn, q_min, q_max, *, schedulers = ("serial", "greedy", "cap sweep", "pipeline", "anneal"),
                  order = None, sa_kwargs = None, seeds = (23,), verbose = False, **kw):
    adj, adj_w = build_adjacency(G)
    if order is None:
        best_w = None
        for sd in seeds:
            o, _e, w, _h, _b = sa_fn(G, **dict(sa_kwargs or {}, seed = sd, qubit_limit = q_max, stop_when_feasible = False))
            if best_w is None or w < best_w:
                order, best_w = o, w
        print(f"one shared order, {best_w} qubits serially\n")

    out = {}
    for name in schedulers:
        out[name] = progressive_depth_search(G, sa_fn, q_min, q_max, scheduler = name, order = order, verbose = verbose, **kw)
        rows = [r for r in out[name]["ramp"] if r["feasible"]]
        if not rows:
            print(f"  {name:<12} nothing feasible in range")
            continue
        shallowest = min(r["depth"] for r in rows)
        first_q = min(r["q"] for r in rows if r["depth"] == shallowest)
        print(f"  {name:<12} best {shallowest:>4} columns at q = {first_q:<4} "
              f"{sum(r['seconds'] for r in rows):>6.1f}s total")
    return out
