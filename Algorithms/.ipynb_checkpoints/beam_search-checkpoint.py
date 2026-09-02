import random
import pyzx as zx
import numpy as np
import fractions
import math
import networkx as nx
from tqdm.auto import tqdm
import heapq

from Algorithms.common import compute_cut_fast, full_order, boundary_size

# +------------------------------------------------------------------------------
# +------------------------------------------------------------------------------
# NOTE: This is oudated, needs to be fixed
# can still be used but needs a different method of addressing
# +------------------------------------------------------------------------------
# +------------------------------------------------------------------------------


# THIS IS A MORE SIMPLIIED VERSION
# NOT SURE HOW TO INCORPORATE THE BFS GIVEN THAT THE GRAPH CAN BE DISCONNECTED :P
# I did it in the SA one but that's only for initial order...
# def greedy_beam(G, positions, start = "I", end = "O", beam_width = 5):
#     # get the nodes besides the input and output
#     # I THINK THAT INPUT AND OUPUT MIGHT BE REALLY IMPORTANT LATER
#     # upon parallelisaiton
#     remaining_init = frozenset(n for n in G.nodes if n not in (start, end))

#     beams = [(0, 0, [start], frozenset({start}), remaining_init)]

#     pbar = tqdm(total = len(remaining_init), desc = "Beam Search", dynamic_ncols = True)

#     # while there are still notes remaining
#     while beams[0][4]:  
#         # initialise a list of candidates
#         candidates = []

#         # go over the beams
#         for peak_b, total_b, order, selected, remaining in beams:
#             # for each reamining node
#             for v in remaining:
#                 # add it to new set
#                 new_selected = selected | {v}
#                 # calculate the boundary size (frontier)
#                 b = boundary_size(G, new_selected)
#                 # get the peak -> old peak (initially 0) vs the new one (frontier size)
#                 new_peak  = max(peak_b, b)  
#                 # calcualte the new total (boundary size)
#                 new_total = total_b + b

#                 # append the candidate
#                 candidates.append((
#                     new_peak,               
#                     new_total,             
#                     order + [v],
#                     new_selected,
#                     remaining - {v},
#                 ))

#         # sort the candidates based on the peak, then boundary
#         candidates.sort(key = lambda x: (x[0], x[1]))
#         # get the first 5 candidates -> if there is more of the same peak (keep all with sam peak)
#         beams = candidates[:beam_width]
#         # for i in beams:
#         #     print(i)
#         # print("DONE")

#         pbar.update(1)
#         pbar.set_postfix(peak = beams[0][0], total = beams[0][1], beams = len(beams))

#     pbar.close()

#     best_order = beams[0][2]
#     best_order.append(end)
#     max_cut, boundaries = compute_cut_fast(G, full_order(best_order))
#     # print(best_order)
#     return best_order, max_cut, boundaries

def vertex_extra(L, R, localize=False):
    L = int(L)
    R = int(R)

    if L == 0 and R == 0:
        return ()

    # Boundary spiders: 0 -> R or L -> 0
    if L == 0 or R == 0:
        degree = max(L, R)

        if degree <= 1:
            return ()

        if R > L:
            return ((-1, degree - 1),)

        # merge / adjoint-cat-state disposal
        if localize:
            return ((0, degree),)

        # full decreasing drain:
        # 3 -> 0 gives ((0, 3), (1, 2))
        # 2 -> 0 gives ((0, 2))
        return tuple((offset, degree - offset) for offset in range(degree - 1))

    # interior spiders
    difference = R - L

    if difference > 0:
        # example: 1 -> 2
        return ((-1, difference),)

    if difference < 0:
        amount = -difference

        if localize:
            return ((0, amount),)

        # for the degree-three cases this gives:
        # 2 -> 1: ((0, 1))
        return tuple((offset, amount - offset) for offset in range(amount))

    return ()

def prepare_graph_data(G):
    # DONE BEFORE I HAD THE WHOLE ADJ MATRIX IN PIPELINE (OUTDATED)
    adjacency = {vertex: tuple((neighbour, G.number_of_edges(vertex, neighbour)) for neighbour in G.neighbors(vertex)) for vertex in G.nodes}
    total_degree = {vertex: sum(amount for _, amount in adjacency[vertex]) for vertex in G.nodes}

    # self-loops affect the spider degree calculation, but they do not cross a graph cut
    # usually this will simply be zero
    loop_degree = {vertex: G.number_of_edges(vertex, vertex) for vertex in G.nodes}

    return adjacency, total_degree, loop_degree


def schedule_vertex_extra(left_degree, right_degree, pending_after_current, localize = False):
    future = list(pending_after_current)
    immediate_extra = 0

    for offset, amount in vertex_extra(left_degree, right_degree, localize = localize):
        if offset == -1:
            immediate_extra += amount
            continue

        if offset < 0:
            raise ValueError(f"Unexpected vertex-extra offset: {offset}")

        missing = offset + 1 - len(future)

        if missing > 0:
            future.extend([0] * missing)

        future[offset] += amount

    return immediate_extra, tuple(future)


def compute_augmented_cut(G, order, start = "I", end = "O", localize = False, prepared = None):
    if prepared is None:
        prepared = prepare_graph_data(G)

    adjacency, total_degree, loop_degree = prepared

    n_boundaries = max(0, len(order) - 1)
    base_cut = np.zeros(n_boundaries, dtype=np.int64)
    extra_cut = np.zeros(n_boundaries, dtype=np.int64)

    selected = set()
    current_cut = 0

    for position, vertex in enumerate(order):
        left_degree = sum(amount for neighbour, amount in adjacency[vertex] if neighbour in selected)
        right_degree = total_degree[vertex] - left_degree

        if vertex != start and vertex != end:
            for offset, amount in vertex_extra(left_degree, right_degree, localize = localize):
                boundary_index = position + offset

                if 0 <= boundary_index < n_boundaries:
                    extra_cut[boundary_index] += amount

        if position < n_boundaries:
            crossing_right = right_degree - loop_degree[vertex]
            current_cut += crossing_right - left_degree
            base_cut[position] = current_cut

        selected.add(vertex)

    augmented_cut = base_cut + extra_cut
    return base_cut, extra_cut, augmented_cut


def greedy_beam(G, start = "I", end = "O", beam_width = 5, localize = False, use_frontier = False, keep_ties = False, show_progress = True):

    prepared = prepare_graph_data(G)
    adjacency, total_degree, loop_degree = prepared
    remaining_init = frozenset(vertex for vertex in G.nodes if vertex != start and vertex != end)
    initial_cut = int(boundary_size(G, {start}))

    # state:
    # (peak_cut, total_cut, order, remaining, current_base_cut, pending_extra)
    # pending_extra[0] contributes to the next finalized boundary
    beams = [(0, 0, (start,), remaining_init, initial_cut, ())]

    pbar = tqdm(total=len(remaining_init), desc="Beam Search", dynamic_ncols=True, disable=not show_progress)

    score_key = lambda candidate: (candidate[0], candidate[1])

    while beams and beams[0][3]:
        candidates = []

        for (peak_cut, total_cut, order, remaining, current_cut, pending_extra) in beams:
            all_choices = []
            frontier_choices = []

            # because every graph vertex is either:
            #   - end,
            #   - still remaining, or
            #   - already selected,
            # we do not need to store and copy a separate selected set
            for vertex in remaining:
                left_degree = sum(amount for neighbour, amount in adjacency[vertex] if neighbour != end and neighbour not in remaining)

                choice = (vertex, left_degree)
                all_choices.append(choice)

                if left_degree > 0:
                    frontier_choices.append(choice)

            if use_frontier and frontier_choices:
                choices = frontier_choices
            else:
                choices = all_choices

            # this contribution was scheduled by vertices selected earlier
            due_extra = pending_extra[0] if pending_extra else 0
            # once the current boundary is consumed, these entries correspond
            # to boundaries after the newly selected vertex
            pending_after_current = (pending_extra[1:] if pending_extra else ())

            for vertex, left_degree in choices:
                right_degree = total_degree[vertex] - left_degree

                immediate_extra, new_pending_extra = (schedule_vertex_extra(left_degree, right_degree, pending_after_current, localize = localize))
                # this is exactly the boundary immediately before vertex
                boundary_cut = (current_cut + due_extra + immediate_extra)

                new_peak = max(peak_cut, boundary_cut)
                new_total = total_cut + boundary_cut

                crossing_right = (right_degree - loop_degree[vertex])
                new_current_cut = (current_cut - left_degree + crossing_right)

                candidates.append((new_peak, new_total, order + (vertex,), remaining.difference((vertex,)), new_current_cut, new_pending_extra))

        if not candidates:
            break

        if len(candidates) <= beam_width:
            candidates.sort(key=score_key)
            beams = candidates

        elif keep_ties:
            candidates.sort(key=score_key)
            cutoff = score_key(candidates[beam_width - 1])
            beams = [candidate for candidate in candidates if score_key(candidate) <= cutoff]

        else:
            # strictly enforce beam_width and avoid sorting every candidate
            beams = heapq.nsmallest(beam_width, candidates, key = score_key)
            beams.sort(key=score_key)

        pbar.update(1)
        pbar.set_postfix(peak=beams[0][0], total=beams[0][1], beams=len(beams))

    pbar.close()

    if not beams:
        return [], 0, [], []

    finalists = []

    for (peak_cut, total_cut, order, remaining, current_cut, pending_extra) in beams:
        final_extra = pending_extra[0] if pending_extra else 0
        final_boundary_cut = current_cut + final_extra

        finalists.append((max(peak_cut, final_boundary_cut), total_cut + final_boundary_cut, order + (end,)))

    best_peak, best_total, best_order_tuple = min(finalists, key = lambda candidate: (candidate[0], candidate[1]))
    best_order = list(best_order_tuple)
    base_cut, extra_cut, augmented_cut = (compute_augmented_cut(G, best_order, start = start, end = end,
                                                                               localize = localize, prepared = prepared))

    max_cut = (int(augmented_cut.max()) if augmented_cut.size else 0)
    _, _, boundaries = compute_cut_fast(G, best_order)

    return best_order, max_cut, augmented_cut, boundaries