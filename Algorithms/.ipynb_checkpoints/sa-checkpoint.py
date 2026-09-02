import random
import pyzx as zx
import numpy as np
import fractions
import math
import networkx as nx

from Algorithms.common import compute_cut, full_order
    
def edge_count(G, u, v):
    # multigrpah can allow more than one parallel edge
    # in some ecxmples it sjust 1 but if there is a paralel, edge it can return >1
    # print(len(G[u][v]))
    if G.has_edge(u, v):
        return len(G[u][v])
    return 0

# update the cut array in-place after swapping positions i and j (i < j)
# cut: array of cut values
# order: ordering of the nodes
# pos: mapping node -> position
# G: the graph
# i, j: positions being swapped
def swap_update_cut(cut, order, pos, G, i, j):
    
    # u, v = order[i], order[j]
    # get their order
    u, v = order[i], order[j]

    def adjust(w, pivot_lo, pivot_hi, delta):
        # increment cut[k] by delta for k for every cut boundary
        # pivot_lo <= k < pivot_hi
        # an edge contributes to every cut boundary between its endpoints
        # so we can just use the range as in the function above 
        # we don't use position now but the upper and lower node we are considering
        # i.e., i and j
        for k in range(pivot_lo, pivot_hi):
            cut[k] += delta
        # print(cut)

    # for node u and node v (and vice versa) (ex. u = 8, v = 7; u = 7, v = 8)
    # we need to handle the effects of moving u and eddects of moving v
    for node, other in [(u, v), (v, u)]:
        # print(node, other)

        # for each of the neighbours of node (examine neighbours)
        for w in G.neighbors(node):
            # ignore the edge between swapped nodes -> swapping endpoints of the sae edge does not change 
            if w == other:
                continue

            # get the position of the neightbour
            pw = pos[w]
            # multiplicity of the edge -> MUTLIGRAPH
            count = edge_count(G, node, w)
            # print(count)
            # moving u from the left to right
            if node == u:
                # neightbour w left of swap range -> more crossings
                # after swap, u moves further to the right
                # therefore, (w, u) crosses more cuts (between i and j)
                if pw < i:
                    adjust(w, i, j, +count) 
                # neighbor left of swap range -> fewer crossngs
                # after swap u moves closer to w
                elif pw > j:
                    adjust(w, i, j, -count)   # w right of swap range → fewer crossings
                # neighbour in between i and j
                # after the swap u jumps to the other side: some cuts stop being crossed, other cuts start being crossed
                else:                          
                    adjust(w, i, pw, -count)
                    adjust(w, pw, j, +count)
            # symmetric logic for v
            # moving v from right ot left
            else:  # node == v
                if pw > j:
                    adjust(w, i, j, +count)
                elif pw < i:
                    adjust(w, i, j, -count)
                else:
                    adjust(w, i, pw, +count)
                    adjust(w, pw, j, -count)

    # return new curwidth after update
    return max(cut) if cut else 0

def local_window_optimize(order, G, window_size = 4):

    best_order = order[:]
    cut, boundaries = compute_cut(full_order(order), G)
    best_cut = max(cut)

    n = len(order)

    # choose random window
    start = random.randint(0, n - window_size)

    window = order[start:start + window_size]

    # try all permutations in the window
    for perm in itertools.permutations(window):

        candidate = (
            order[:start]
            + list(perm)
            + order[start + window_size:]
        )

        cw = max(compute_cut(full_order(candidate), G))

        if cw < best_cut:
            best_cut = cw
            best_order = candidate

    return best_order, best_cut

# def compute_energy(cut, w_max = 1.0, w_sum = 0.1, w_var = 0.05):

#     if not cut:
#         return 0

#     m = max(cut)
#     s = sum(cut)

#     # mean = s / len(cut)

#     var = sum((x - s/len(cut)) ** 2 for x in cut) / len(cut)

#     # ANDREYS IDEA ON USING DIFFERENT VAIRABLES as a cost function essentially
#     return (w_max * m + w_sum * s + w_var * var)

# WHAT BEN SAID ABOUT USING VARIANCE AND STUFF FOR THE FUNCITON
    # w_max - weight given to maximum cut
    # w_sum - weight given to the sum of cuts
    # w_var - weight given to the variation of the cut
def compute_energy(cut, w_max = 1.0, w_sum = 0.1, w_var = 0.05):
    if len(cut) == 0:
        return 0.0

    arr = np.asarray(cut)

    return (w_max * arr.max() + w_sum * arr.sum() + w_var * arr.var())

def init_order(G, vertices):
    # pin I first, O last —> only optimise internal nodes (input and output do matter, even with OCM)
    internal = [v for v in vertices if v not in ("I", "O")]
    n_internal = len(internal)
    # if there a re no internal nodes, jsut return input and output, with no cut
    if n_internal == 0:
        return ["I", "O"], 0, []

    # start is the node with the maximum degree (if there is a tie, first one)
    # given the nodes and their degrees -> (ndoe, degree)
    # print(G.subgraph(internal).degree())
    # print(max(G.subgraph(internal).degree(), key = lambda x: x[1]))
    start = max(G.subgraph(internal).degree(), key = lambda x: x[1])[0]
    # get the maximum degree (second entry in the touple)

    # EXPLORE DFS TOO ???
    # dfs_order = list(nx.dfs_tree(G.subgraph(internal), start).nodes())
    # visited = set(dfs_order)
    # for v in internal:
    #     if v not in visited:
    #         dfs_order.append(v)
    # return dfs_order, n_internal

    # get the initial order (BFS)
    bfs_order = list(nx.bfs_tree(G.subgraph(internal), start).nodes())
    # get the set of visited nodes
    visited = set(bfs_order)
    # if a ndoe is not visited, just append it
    # MAYBE RUN BFS AGAIN ON THE UNVISITED NOTES????
    for v in internal:
        if v not in visited:
            bfs_order.append(v)
    return bfs_order, n_internal

def simulated_annealing_cutwidth(G, T_init = None, T_min = 1e-4, alpha = 0.9, 
                                 steps_per_temp = None, seed = None, prob_adj = 0.7):
    if seed is not None:
        random.seed(seed)

    # get the set of vetives and the number of them
    vertices = list(G.nodes())
    n = len(vertices)

    # if there are no nodes, return empty stuff
    if n == 0:
        return [], 0, []

    # set the initial order (without input/oputpu nodes) and get the the lenth of it
    order, n_internal = init_order(G, vertices)
    steps_per_temp = steps_per_temp or 15 * n_internal

    # get the position of the nodes (WITH I/O) -> just to preserve final order
    pos = {v: i for i, v in enumerate(full_order(order))}
    # get the inital cut
    cut, boundaries = compute_cut(full_order(order), G)
    # get the "energy" -> maximum cut (needs to be reduced)
    energy = compute_energy(cut) if cut else 0

    # set the initial temerature 
    # NEED TO DO MORE RESEARCH ON THIS
    T_init = T_init or max(energy * 0.5, 1.0)

    # best parameters are initiall the inital parameters
    best_order = order[:]
    # print(best_order)

    best_energy = energy
    T = T_init

    # to store the history
    history = []
    pos_history = []
    best_history = {}
    iteration = 0

    # while T is > minimum temperature
    while T > T_min:
        
        # for the number of timesteps in the current temperature
        for _ in range(steps_per_temp):
            # -----------------------------------------------------------------------
            # # get the ith and jth node (from the internal nodes)
            # # do this RANDOMLY in SA
            # i, j = sorted(random.sample(range(n_internal), 2))
            # # i = random.randrange(n_internal - 1)
            # # j = i + 1

            # # get the full order
            # fo = full_order(order)
            
            # # shift i/j by 1 to account for pinned "I" at position 0
            # fi, fj = i + 1, j + 1

            # # get the trial cut (initially the initial cut)
            # trial_cut  = cut[:]
            # # new energy -> maximum cut of the swap of i an j
            # new_energy = swap_update_cut(trial_cut, fo, pos, G, fi, fj)
            # # get the change in energy
            # delta = new_energy - energy
            # -----------------------------------------------------------------------

            move_type = random.random()

            # RANDOMLY 
            # either just take adjecent nodes (PREFER THIS so higher prob)
            # OR
            # any two random nodes
            if move_type < prob_adj:
                i = random.randrange(n_internal - 1)
                j = i + 1
            
                fo = full_order(order)
                # shift i/j by 1 to account for pinned "I" at position 0
                fi, fj = i + 1, j + 1

                # get the candidate order
                candidate_order = order[:]
                candidate_order[i], candidate_order[j] = (candidate_order[j], candidate_order[i])

            else:
                # randomly select indices i and j 
                i, j = sorted(random.sample(range(n_internal), 2))
                candidate_order = order[:]
                node = candidate_order.pop(i)
                candidate_order.insert(j, node)

            # update the candidate cut
            candidate_cut, boundaries = compute_cut(full_order(candidate_order), G)
            # pdate energy
            new_energy = compute_energy(candidate_cut)
            # caculate the change
            delta = new_energy - energy

            # if the delta is < 0 
            # -> this means that we are decreasing the workload or the number of edges crossing the range (u, w)
            #    -> see the swap function that calculates the cut
            # or math.exp(-delta / T) greater than a random number in (0,1) -> this is the "random step"
            # -> we can randomly choose even a worse option to escape the minimum
            # -> if I add more weight to the denumerator -> the order changes slightly 
            #    -> still same number of qubits BUT prevents reuse of ancilae?
            # -----------------------------------------------------------------------
            # if delta < 0 or random.random() < math.exp(-delta / (T)):   
            #     # swap the nodes
            #     order[i], order[j] = order[j], order[i]
            #     # update the positions
            #     pos = {v: k for k, v in enumerate(full_order(order))}
            #     # print(pos)
            #     # update the cut
            #     cut    = trial_cut
            #     # and update the enrgy
            #     energy = new_energy

            #     # if the energy is lower -> better
            #     if energy < best_energy:
            #         # update the enrgy and update the best order
            #         best_energy = energy
            #         best_order  = order[:]
            #         pos_history.append({round(T, 3): pos})
            # -----------------------------------------------------------------------
            if delta < 0 or random.random() < math.exp(-delta / T):
                order = candidate_order
                cut = candidate_cut
                energy = new_energy
            
                pos = {v: k for k, v in enumerate(full_order(order))}
                history.append(pos)
            
                if energy < best_energy: 
                    best_energy = energy
                    best_order = order[:]
                    best_history[T] = {"ord": best_order, "eng": best_energy}
        T *= alpha

        cut, boundaries = compute_cut(full_order(best_order), G)
        max_cut = max(cut)
                    
    return ["I"] + best_order + ["O"], best_energy, max_cut, history, best_history
    # return ["I"] + best_order + ["O"], best_energy, history, pos_history