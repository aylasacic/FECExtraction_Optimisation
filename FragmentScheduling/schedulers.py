import math
import random

from .fragments import NotExtractable
from .layers import greedy_layers, leg_directions_ok
from .occupancy import occupancy_profile

__all__ = [
    "sched_serial",
    "sched_asap",
    "sched_capped",
    "sched_cap_sweep",
    "sched_anneal"
]


def _profile(G, sched, adj_w, q):
    try:
        return occupancy_profile(G, sched, adj_w, q)
    except NotExtractable:
        return None


def sched_serial(G, order, adj, adj_w, q, **kw):
    return [[v] for v in order]


def sched_asap(G, order, adj, adj_w, q, **kw):
    return greedy_layers(G, order, adj=adj)


def sched_capped(G, order, adj, adj_w, q, cap=2, **kw):
    layers, sets = [], []
    for v in order:
        placed = False
        for j, s in enumerate(sets):
            if len(s) < cap and not (adj[v] & s):
                layers[j].append(v)
                s.add(v)
                placed = True
                break
        if not placed:
            layers.append([v])
            sets.append({v})
    return [s for s in layers if s]


def sched_cap_sweep(G, order, adj, adj_w, q, **kw):
    best = None
    for cap in range(1, len(order) + 1):
        sched = sched_capped(G, order, adj, adj_w, q, cap=cap)
        prof = _profile(G, sched, adj_w, q)
        if prof is None or not prof.feasible:
            continue
        if best is None or prof.depth < best[0]:
            best = (prof.depth, sched)
        if len(sched) <= 1:
            break
    return best[1] if best else sched_serial(G, order, adj, adj_w, q)


def sched_anneal(G, order, adj, adj_w, q, budget = 98, seed = 0, start = None, lam = 4.0, mu = 0.05,
                 T_init = 1.5, T_min = 0.02, alpha = 0.99):

    rng = random.Random(seed)
    sched = [list(s) for s in (start or sched_cap_sweep(G, order, adj, adj_w, q))]

    def energy(prof):
        over = sum((a - q) ** 2 for a in prof.occ if a > q)
        return prof.depth + lam * over + mu * prof.volume / max(q, 1)

    prof = _profile(G, sched, adj_w, q)
    if prof is None:
        return None
    cur = energy(prof)
    best = (sched, prof) if prof.feasible else None

    def fits(v, layer):
        return not (adj[v] & set(layer))

    n_temps = max(1, math.ceil(math.log(T_min / T_init) / math.log(alpha)))
    per = max(1, budget // n_temps)
    T = T_init
    for _ in range(n_temps):
        for _ in range(per):
            move = rng.random()
            cand = [list(s) for s in sched]
            # relocate
            if move < 0.40 and len(cand) > 1:
                src = rng.randrange(len(cand))
                v = rng.choice(cand[src])
                tgt = rng.randrange(len(cand) + 1)
                if tgt == src:
                    continue
                cand[src].remove(v)
                if tgt >= len(cand):
                    cand.append([v])
                elif fits(v, cand[tgt]):
                    cand[tgt].append(v)
                else:
                    continue
            # merge a layer
            elif move < 0.70 and len(cand) > 1:
                t = rng.randrange(len(cand) - 1)
                pool = cand[t] + cand[t + 1]
                rng.shuffle(pool)
                keep, out = [], []
                for v in pool:
                    (keep if fits(v, keep) else out).append(v)
                cand = cand[:t] + [keep] + cand[t + 2:]
                ok = True
                for v in out:
                    js = [j for j in range(len(cand)) if j != t]
                    rng.shuffle(js)
                    for j in js:
                        if fits(v, cand[j]):
                            cand[j] = cand[j] + [v]
                            break
                    else:
                        ok = False
                        break
                if not ok:
                    continue
            # split a layer
            elif len(cand) > 0:
                fat = [t for t, s in enumerate(cand) if len(s) >= 2]
                if not fat:
                    continue
                t = rng.choice(fat)
                s = cand[t][:]
                rng.shuffle(s)
                k = rng.randrange(1, len(s))
                cand = cand[:t] + [s[:k], s[k:]] + cand[t + 1:]
            cand = [s for s in cand if s]

            if not leg_directions_ok(cand):
                continue
            cp = _profile(G, cand, adj_w, q)
            if cp is None:
                continue
            ce = energy(cp)
            if ce - cur > 0 and rng.random() >= math.exp(-(ce - cur) / T):
                continue
            sched, prof, cur = cand, cp, ce
            if cp.feasible and (best is None or (cp.depth, cp.volume) < (best[1].depth, best[1].volume)):
                best = (cand, cp)
        T *= alpha
    return best[0] if best else None
