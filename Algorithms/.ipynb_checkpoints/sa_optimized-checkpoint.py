import random
import pyzx as zx
import numpy as np
import fractions
import math
import networkx as nx
from tqdm.auto import tqdm

from Algorithms.common import compute_cut_fast, full_order
from Utils.Computation.compute_cuts import init_order

from GraphGeneration.three_ary_graph import random_3ary_zx_graph
from GraphGeneration.random_graph import random_zx_graph
from GraphGeneration.zx_to_nx import nx_graph, print_graph, collapse_io, split_vertex
from Algorithms.sa import *

from Algorithms.common import compute_cut_fast


from Algorithms.depth_reduction import progressive_depth_search
from Utils.ordering import order_nodes, order_nodes_graphing
from Utils.print_OLA_graphs import plot, schedule_to_pos

# from CircuitExtraction.set_attributes import apply_targetmapper_patches 
# from CircuitExtraction.circuit_extractor import CircuitQubitAllocator, CircuitExtractor
# from CircuitExtraction.graph_extractor import GraphQubitAllocator, GraphExtractor


def compute_cut_matrix(order, A, node_to_idx):
    n = len(order)
    if n <= 1:
        return np.zeros(0, dtype=np.int64), []
    # node_to_index -> done outside
    idx = [node_to_idx[v] for v in order]
    # permutes the rows and columns of A into the current ordering
    B = A[np.ix_(idx, idx)]

    # empty n-1 by n-1 matrix
    cut = np.zeros(n - 1, dtype=np.int64)
    # intial case (bsae case)
    # the first boundary sits right after v_0, so the left block is just {v_0}
    # every edge from v_0 to anyone else crosses it
    # we just nsum up the entries for row 0 from column 1 onwards
    cut[0] = int(B[0, 1:].sum())

    # insead or resumming a whole submatrix for each boundary, we derive cut[k] from cut[k-1]
    # moving the dividing line cahnges the sude of exactly one vertex (if going in step 1)
    # so at k-1 | k -> moving it -> v_k moves from right to left block and everything else stays put
    for k in range(1, n - 1):
        # gained edges are from v_k to vertices further right {v_{k+1},...,v_{n-1}}
        gain = int(B[k, k + 1:].sum())
        # lost are the edges from v_k to vertices to its left {v_0,...,v_{k-1}}
        # since now botha reon the left, they stop crossing
        lose = int(B[:k, k].sum())
        # all others keep their status unchanged
        cut[k] = cut[k - 1] + gain - lose

    boundaries = [(order[i], order[i + 1]) for i in range(n - 1)]
    return cut, boundaries

# bund;e everything the search needs into a dictionary
def make_cut_cache(G, order):
    nodes = list(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist = nodes, dtype = np.int64, weight = None)
    cut, boundaries = compute_cut_matrix(order, A, node_to_idx)
    return {"A": A, "node_to_idx": node_to_idx, "order": order[:], "pos": {v: i for i, v in enumerate(order)}, 
            "cut": cut, "boundaries": boundaries}

# ---- delta/apply split so SA can evaluate a move without copying the cache ----
# these compute the change a move would make, and separately apply it in place.
# compute_adjacent_swap / delta_remove_insert (below) stay as mutating wrappers.

def adjacent_swap_delta(cache, i):
    # returns the signed change to cut[i] from swapping order[i], order[i+1].
    # only cut[i] changes; a-b crosses in both layouts so it cancels out.
    A = cache["A"]; node_to_idx = cache["node_to_idx"]; order = cache["order"]
    ia = node_to_idx[order[i]]
    ib = node_to_idx[order[i + 1]]
    left_idx = [node_to_idx[v] for v in order[:i]]
    right_idx = [node_to_idx[v] for v in order[i + 2:]]
    old_cross = new_cross = 0
    if left_idx:
        old_cross += A[left_idx, ib].sum()
        new_cross += A[left_idx, ia].sum()
    if right_idx:
        old_cross += A[ia, right_idx].sum()
        new_cross += A[ib, right_idx].sum()
    return int(new_cross - old_cross)

def remove_insert_positions(cache, i, j):
    # apply the move to order/pos and return (lo, hi) window whose cuts changed
    order = cache["order"]; pos = cache["pos"]
    node = order.pop(i)
    order.insert(j, node)
    lo, hi = min(i, j), max(i, j)
    for k in range(lo, hi):
        pos[order[k]] = k
    return lo, hi

# pulls a node out of position i and reinserts it at j
def delta_remove_insert(cache, i, j):
    # mutating wrapper kept for API compatibility.
    if i == j:
        return cache["cut"]
    lo, hi = remove_insert_positions(cache, i, j)
    recompute_cut_interval(cache, lo, hi)
    return cache["cut"]

def recompute_cut_interval(cache, out_cut, i, j):
    # like recompute_cut_interval, but computes the post-move cut values into
    # out_cut, using a hypothetical order where order[i] is removed and
    # reinserted at j. does NOT mutate cache. uses the same telescoping recurrence.
    A = cache["A"]; node_to_idx = cache["node_to_idx"]; order = cache["order"]
    n = len(order)
    if n <= 1:
        return out_cut

    # hypothetical order after the move (local list, cache untouched)
    new_order = order[:]
    node = new_order.pop(i)
    new_order.insert(j, node)

    lo = max(0, min(i, j))
    hi = min(n - 2, max(i, j))
    idx = [node_to_idx[v] for v in new_order]

    left_idx = idx[:lo + 1]
    right_idx = idx[lo + 1:]
    out_cut[lo] = int(A[np.ix_(left_idx, right_idx)].sum()) if right_idx else 0

    for k in range(lo + 1, hi + 1):
        ik = idx[k]
        gain = int(A[ik, idx[k + 1:]].sum()) if k + 1 <= n - 1 else 0
        lose = int(A[idx[:k], ik].sum())
        out_cut[k] = out_cut[k - 1] + gain - lose
    return out_cut

# pulls a node out of position i and reinserts it at j
def delta_remove_insert(cache, i, j):
    order = cache["order"]
    pos = cache["pos"]
    n = len(order)

    if i == j:
        return cache["cut"]

    # mutates the list: removes element at i
    node = order.pop(i) 
    # mutates the list: reinserts it at j
    order.insert(j, node) 

    lo = min(i, j)
    hi = max(i, j)

    # every boundary between i and j has its partition changes
    # so update position in the order
    for k in range(lo, hi):
        # mutates the dict: rewrites position entries
        pos[order[k]] = k

    # exploratory move
    recompute_cut_interval(cache, lo, hi)

    return cache["cut"]

# ENERGY FUNCTION
# ----------------------------------------------------------------------
# COMBINED
# ----------------------------------------------------------------------
def total_energy(cut_arr, qubit_limit, lam = 1000, p = 8):
    arr = np.asarray(cut_arr, dtype = np.float64)
    if arr.size == 0:
        return 0.0
    m = arr.max()
    base = 0.0 if m == 0 else m * (np.mean((arr / m) ** p)) ** (1.0 / p)
    over = np.maximum(0.0, arr - qubit_limit)
    return base + lam * (over ** 2).sum()
# ---------------------------------------------------------------------- 

def simulated_annealing_feasibility(G, qubit_limit = 98, T_init = None, T_min = 1e-4, alpha = 0.995, 
                                    steps_per_temp = None, seed = 23, prob_adj = 0.7, lam = 1000, stop_when_feasible = False):
    if seed is not None:
        random.seed(seed)

    vertices = list(G.nodes())
    n = len(vertices)

    if n == 0:
        return [], 0, 0, [], {}

    order, n_internal = init_order(G, vertices)

    if n_internal <= 1:
        full_current_order = full_order(order)
        cache = make_cut_cache(G, full_current_order)

        cut = cache["cut"]
        max_cut = int(cut.max()) if len(cut) else 0
        energy = ordering_energy(cut)

        return full_current_order, energy, max_cut, [], {}

    steps_per_temp = steps_per_temp or 4 * n_internal

    # build initial state 
    # ------------------------------------------------------
    full_current_order = full_order(order)
    cache = make_cut_cache(G, full_current_order)

    cut = cache["cut"]
    current_max_cut = int(cut.max()) if len(cut) else 0
    current_feasible = current_max_cut <= qubit_limit
    # ------------------------------------------------------

    # set/calculate initial paremeters
    # ------------------------------------------------------
    # energy = ordering_energy(cut)
    # energy = total_energy(cut, qubit_limit, lam)
    energy = total_energy(cut, qubit_limit)

    T_init = T_init or max(energy * 0.5, 1.0)
    T = T_init

    best_order = order[:]
    best_energy = energy
    best_max_cut = current_max_cut

    best_feasible_order = order[:] if current_feasible else None
    best_feasible_energy = energy if current_feasible else None
    best_feasible_max_cut = current_max_cut if current_feasible else None
    
    history = []
    best_history = {}
    iteration = 0

    total_temps = math.ceil(math.log(T_min / T_init) / math.log(alpha))
    pbar = tqdm(total = total_temps, desc = "SimAnneal",  unit = "temp", dynamic_ncols = True)
    # ------------------------------------------------------
    
    while T > T_min:
        accepted = 0
        # for each temperature iterate over the steps
        for _ in range(steps_per_temp):
            # evaluate a trial move WITHOUT copying the whole cache. we compute the
            # candidate cut array cheaply, decide accept/reject, then mutate `cache`
            # in place only on accept. move_kind records how to apply it.
            # ------------------------------------------------------
            if random.random() < prob_adj:
                # adjacent swap: only cut[i+1] changes
                i = random.randrange(n_internal - 1)
                si = i + 1
                dcut = adjacent_swap_delta(cache, si)
                if dcut == 0:
                    candidate_cut = cut  # unchanged; reuse without allocating
                else:
                    candidate_cut = cut.copy()
                    candidate_cut[si] += dcut
                move_kind = ("swap", si, dcut)
            else:
                # remove/insert: recompute only the affected window on a copy
                i, j = sorted(random.sample(range(n_internal), 2))
                si, sj = i + 1, j + 1
                candidate_cut = cut.copy()
                recompute_cut_interval(cache, candidate_cut, si, sj)
                move_kind = ("move", si, sj)

            candidate_max_cut = int(candidate_cut.max()) if len(candidate_cut) else 0
            candidate_feasible = candidate_max_cut <= qubit_limit

            # new_energy = total_energy(candidate_cut, qubit_limit, lam)
            new_energy = total_energy(candidate_cut, qubit_limit)
            # ------------------------------------------------------

            # accept/reject (identical logic to before)
            if not current_feasible:
                old_repair_energy = energy
                new_repair_energy = new_energy
                delta_repair = new_repair_energy - old_repair_energy
                                                                # Boltzmann probability distribution
                '''
                    Boltzmann probability distribution,
                    Prob (E)∼ exp(−E/kT ) (10.9.1)
                    expresses the idea that a system in thermal equilibrium at temperature T has its
                    energy probabilistically distributed among all different energy states E. Even at
                    low temperature, there is a chance, albeit very small, of a system being in a high
                    energy state. Therefore, there is a corresponding chance for the system to get out of
                    a local energy minimum in favor of finding a better, more global, one. The quantity
                    k (Boltzmann’s constant) is a constant of nature that relates temperature to energy.
                    In other words, the system sometimes goes uphill as well as downhill; but the lower
                    the temperature, the less likely is any significant uphill excursion
                    FROM: https://people.sc.fsu.edu/~inavon/5420a/Recipes10_9.pdf
                '''
                accept = (delta_repair < 0 or random.random() < math.exp(-delta_repair / T))
            elif not candidate_feasible:
                accept = False
            else:
                delta = new_energy - energy
                accept = (delta < 0 or random.random() < math.exp(-delta / T))

            # apply the move to the real cache only if accepted
            # -------------------------------------------------------------------------------------
            if accept:
                if move_kind[0] == "swap":
                    _, si, dcut = move_kind
                    a, b = cache["order"][si], cache["order"][si + 1]
                    cache["order"][si], cache["order"][si + 1] = b, a
                    cache["pos"][a], cache["pos"][b] = si + 1, si
                else:
                    _, si, sj = move_kind
                    remove_insert_positions(cache, si, sj)
                cache["cut"] = candidate_cut

                order = cache["order"][1:-1]
                cut = candidate_cut
                energy = new_energy
                current_max_cut = candidate_max_cut
                current_feasible = candidate_feasible
                accepted += 1

                pos = {v: k for k, v in enumerate(cache["order"])}
                history.append(pos)

                if energy < best_energy:
                    best_energy = energy
                    best_order = order[:]
                    best_max_cut = current_max_cut

                    best_history[iteration] = {"T": T, "ord": best_order[:],
                                               "eng": best_energy, "max_cut": best_max_cut}

                if current_feasible:
                    if best_feasible_order is None or energy < best_feasible_energy:
                        best_feasible_order = order[:]
                        best_feasible_energy = energy
                        best_feasible_max_cut = current_max_cut

                    if stop_when_feasible:
                        pbar.close()
                        return (full_order(order), energy, current_max_cut, history, best_history)
            # -------------------------------------------------------------------------------------
        # lowe the temperature
        T *= alpha
        # increment the iteration (for bookkeeping up)
        iteration += 1

        display_energy = (best_feasible_energy if best_feasible_energy is not None else best_energy)
        
        pbar.set_postfix({
            "best_cut": (best_feasible_max_cut if best_feasible_max_cut is not None else best_max_cut),
            "limit": qubit_limit,
            "feasible": best_feasible_order is not None,
            "energy": round(display_energy, 4),
            "T": f"{T:.4g}",
            "accepted": accepted
        })

        pbar.update(1)

    pbar.close()

    # if an order was found, return it
    if best_feasible_order is not None:
        return (full_order(best_feasible_order), best_feasible_energy, best_feasible_max_cut, history, best_history)
    # orderwise just find the last order found
    return (full_order(best_order), best_energy, best_max_cut, history, best_history)