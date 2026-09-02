import math
import random

import numpy as np
from tqdm.auto import tqdm

from Algorithms.sa import full_order, init_order

from .order_cache import (
    commit_order,
    make_occ_cache,
    propose_order_move,
    propose_order_swap,
    validate_occ_cache,
)
from .primitives import total_energy

__all__ = ["simulated_annealing_feasibility"]


def simulated_annealing_feasibility(G, qubit_limit=98, T_init=None, T_min=1e-4, alpha=0.995, steps_per_temp=None,
                                    seed=23, prob_adj=0.7, lam=1000, stop_when_feasible=False, localize=False,
                                    validate_cache=False, validation_frequency=100, keep_history=False, progress=True):
    # anneal a vertex order under a qubit budget

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    vertices = list(G.nodes())
    if not vertices:
        return [], 0.0, 0, [], {}

    order, n_internal = init_order(G, vertices)
    cache = make_occ_cache(G, full_order(order))

    def state():
        occ = cache["occ"]
        peak = int(occ.max()) if occ.size else 0
        return peak, total_energy(occ, qubit_limit, lam=lam)

    peak, energy = state()

    if n_internal <= 1:
        return cache["order"], energy, peak, [], {}

    if steps_per_temp is None:
        steps_per_temp = 4 * n_internal
    if T_init is None:
        T_init = max(energy * 0.5, 1.0)
    T = float(T_init)

    feasible = peak <= qubit_limit
    best = {"order": cache["order"][1:-1], "energy": energy, "peak": peak}
    best_feasible = dict(best) if feasible else None

    history, best_history = [], {}
    accepted_total = 0

    if stop_when_feasible and feasible:
        return full_order(best["order"]), energy, peak, history, best_history

    total_temps = 0
    if T_init > T_min and 0 < alpha < 1:
        total_temps = max(math.ceil(math.log(T_min / T_init) / math.log(alpha)), 0)
    pbar = tqdm(total=total_temps, desc="SimAnneal", unit="temp",
                dynamic_ncols=True, disable=not progress)

    iteration = 0
    while T > T_min:
        accepted_here = 0

        for _ in range(steps_per_temp):
            if n_internal > 1 and random.random() < prob_adj:
                prop = propose_order_swap(cache, random.randrange(n_internal - 1))
            else:
                a, b = random.sample(range(n_internal), 2)
                prop = propose_order_move(cache, a, b)

            cand_peak = prop.peak
            cand_feasible = cand_peak <= qubit_limit
            cand_energy = total_energy(prop.occ, qubit_limit, lam=lam)

            # once feasible, never step back out of the feasible region
            if feasible and not cand_feasible:
                accept = False
            else:
                delta = cand_energy - energy
                accept = delta < 0 or random.random() < math.exp(-delta / T)

            if not accept:
                continue

            commit_order(cache, prop)
            energy, peak, feasible = cand_energy, cand_peak, cand_feasible
            accepted_here += 1
            accepted_total += 1
            internal = list(cache["internal"])

            if keep_history:
                history.append({v: p for p, v in enumerate(cache["order"])})

            if energy < best["energy"]:
                best = {"order": internal[:], "energy": energy, "peak": peak}
                if keep_history:
                    best_history[iteration] = {"T": T, "ord": internal[:],
                                               "eng": energy, "max_cut": peak}

            if feasible and (best_feasible is None or energy < best_feasible["energy"]):
                best_feasible = {"order": internal[:], "energy": energy, "peak": peak}
                if stop_when_feasible:
                    pbar.close()
                    return (full_order(internal), energy, peak, history, best_history)

            if (validate_cache and validation_frequency > 0
                    and accepted_total % validation_frequency == 0):
                validate_occ_cache(cache, G)

        T *= alpha
        iteration += 1

        shown = best_feasible or best

        pbar.set_postfix({"best_cut": shown["peak"], "limit": qubit_limit,
                          "feasible": best_feasible is not None,
                          "energy": round(shown["energy"], 4),
                          "T": f"{T:.4g}", "accepted": accepted_here})
        pbar.update(1)

    pbar.close()
    out = best_feasible or best
    return (full_order(out["order"]), out["energy"], out["peak"], history, best_history)
