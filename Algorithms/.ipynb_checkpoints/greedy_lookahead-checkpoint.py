import random
import pyzx as zx
import numpy as np
import fractions
import math
import networkx as nx
from functools import lru_cache
from tqdm.auto import tqdm


from Algorithms.common import compute_cut, full_order, boundary_size, compute_cut_fast


# O(# of unique states up to depth 5)
# for 3-ary: 1+3+3^2+3^3+3^4+3^5 = 364

'''
BIT MASKING
- set_of_nodes_placed_so_far is the core state that needs to be tracked, hashed, and manipulated efficiently 
- bit masking -> represents this set as a single integer
    - each internal node is assigned a unique power of two
      EXAMPLE:
      A → 0001
      B → 0010
      C → 0100
      D → 1000
      and a set like {A, C} becomes the integer 0101 = 5...

OPERATIONS:
- add node v: mask | bit_of[v]
- all reamining: all_mask ^ mask -> XOR fplits placed bits to 0
- is v plced?: mask & bit_of[v] -> non-zero if bit i set
- all placed?: mask == all_mask -> all bits set

iterate the remaning nodes using the standard python trick: mask & -mask
    - flip all bits and add 1
    - this isolated the lowest set bit
    - XORING it back clears it
    - iterating over exactly the bits that are set -> no branching ber bit

NOTES:
- cache key for best_k_score is (mask, depth) —> two integers
    - because the mask enocdes the set of placed nodes and not their order, 
        - any two paths through the search tree that place the same nodes end up at the same cache entry
        - wihtout this, the tree would be re-explored exponentially. 
        - with it -> the # of unique states is at most 2^n × k (I THINK)
        - BVUT far smaller than the number of orderings: n!

RUNTIME:

O(∣masks visited∣) -> lru_cache


                    mc_k_score                                             beam_k_score
strategy            random rollouts, greedy steps                          deterministic/systematic expansion + pruning
branching           1 greedy choice/step                                   many candidates per step
diversity           via random sampling                                    via keeping top-k paths
optimism            min over rollouts                                      min over beam survivors
cost                depends on num_rollouts * depth evaluations            depends on beam_width * branch_pool_size * depth

'''

def greedy_lookahead_mc(G, positions, start = "I", end = "O", k = 4, lookahead_pool_size = 100, lookahead_sample_size = 25, 
                        top_level_pool_size = None, seed = 42):
    rng = random.Random(seed)
    internal = [v for v in G.nodes if v not in (start, end)]
    internal.sort(key = lambda v: (positions[v][0], positions[v][1], str(v)))

    n = len(internal)

    if n == 0:
        final_order = [start, end]
        max_cut, cut, boundaries = compute_cut_fast(G, final_order)
        return final_order, cut, boundaries

    bit_of = {v: 1 << i for i, v in enumerate(internal)}
    node_of = {1 << i: v for i, v in enumerate(internal)}
    all_mask = (1 << n) - 1

    edge_to_start = {v: 0 for v in internal}
    edge_to_end = {v: 0 for v in internal}
    internal_adj = {v: {} for v in internal}

    def add_internal_edge(a, b):
        internal_adj[a][b] = internal_adj[a].get(b, 0) + 1

    for u, v, _ in G.edges(keys = True):
        if u == v:
            continue

        if u in bit_of and v in bit_of:
            add_internal_edge(u, v)
            add_internal_edge(v, u)

        elif u in bit_of and v == start:
            edge_to_start[u] += 1
        elif v in bit_of and u == start:
            edge_to_start[v] += 1

        elif u in bit_of and v == end:
            edge_to_end[u] += 1
        elif v in bit_of and u == end:
            edge_to_end[v] += 1

    def iter_bits(mask):
        while mask:
            bit = mask & -mask
            yield bit
            mask ^= bit

    @lru_cache(maxsize = None)
    def boundary_after_mask(mask):
        if mask == 0:
            count = 0

            for u, v, _ in G.edges(keys=True):
                if u == v:
                    continue

                if (u == start) != (v == start):
                    count += 1

            return count

        added_bit = mask & -mask
        prev_mask = mask ^ added_bit

        v = node_of[added_bit]
        old_boundary = boundary_after_mask(prev_mask)

        edges_to_selected = edge_to_start[v]
        edges_to_unselected = edge_to_end[v]

        for nbr, multiplicity in internal_adj[v].items():
            nbr_bit = bit_of[nbr]

            if prev_mask & nbr_bit:
                edges_to_selected += multiplicity
            else:
                edges_to_unselected += multiplicity

        return old_boundary - edges_to_selected + edges_to_unselected

    def candidate_bits_by_immediate(mask, pool_size=None):
        # scores every remaining node by its immediate boundary effect and keeps only the best pool_size of them
        # (optionaly) keep only the best pool_size candidates
        remaining_mask = all_mask ^ mask
        scored = []

        for bit in iter_bits(remaining_mask):
            v = node_of[bit]
            new_mask = mask | bit
            immediate = boundary_after_mask(new_mask)

            scored.append((immediate, positions[v][0], positions[v][1], str(v), bit))

        scored.sort()

        if pool_size is not None:
            scored = scored[:pool_size]

        return [item[-1] for item in scored]

    def sampled_candidate_bits(mask, pool_size, sample_size):
        # keep a promising pool by immediate score, then randomly sample from it
        # essentaly, nodes which look terrible immediately are unlikely to be redeemed by future steps -> discarded cheaply
        pool = candidate_bits_by_immediate(mask, pool_size=pool_size)

        if sample_size is None or sample_size >= len(pool):
            return pool

        # from the surviving candidates only sample_size are drawn u.a.r.
        return rng.sample(pool, sample_size)

    @lru_cache(maxsize = None)
    def best_k_score(mask, depth):
        # SEE COMMENTED CODE UNDERNEATH
        # loosely, given the nodes already placed (encoded in mask),
        # hat is the best worst-case boundary achieved over the next k steps (depth)
        if depth == 0 or mask == all_mask:
            return 0

        best = None

        candidate_bits = sampled_candidate_bits(mask, pool_size = lookahead_pool_size, sample_size = lookahead_sample_size)
        # candidate_bits = candidate_bits_by_immediate(mask, pool_size = lookahead_pool_size)

        for bit in candidate_bits:
            new_mask = mask | bit

            immediate = boundary_after_mask(new_mask)
            future = best_k_score(new_mask, depth - 1)

            worst = max(immediate, future)

            if best is None or worst < best:
                best = worst

        return best if best is not None else 0

    def beam_k_score(mask, depth, beam_width = 50, branch_pool_size = 100, branch_sample_size = None):

        # approximate k-step lookahead using beam search
        # state score is the maximum boundary seen along the partial future
        # lower is better
    
        beam = [(0, mask)]  # (max_boundary_seen_in_future, current_mask)
    
        for _ in range(depth):
            next_beam = []

            # for every state currently in the beam -> branches out by trying candidate bits
            for path_max, cur_mask in beam:
                if cur_mask == all_mask:
                    next_beam.append((path_max, cur_mask))
                    continue

                # two candidate selections modes depending on the branch sampe size
                if branch_sample_size is None:
                    # determinisitc
                    candidate_bits = candidate_bits_by_immediate(cur_mask, pool_size = branch_pool_size)
                else:
                    # random sample
                    candidate_bits = sampled_candidate_bits(cur_mask, pool_size = branch_pool_size,
                                                            sample_size = branch_sample_size)
                    
                # for each child, record the running worst boundary along the path
                for bit in candidate_bits:
                    new_mask = cur_mask | bit
                    immediate = boundary_after_mask(new_mask)
                    new_path_max = max(path_max, immediate)
                    # keep the beam narrow here
                    # through some pruning
                    next_beam.append((new_path_max, new_mask))

            if not next_beam:
                break

            # after explanding all states, it sorts by the maiximm path
            # as before, lower is better
            # keep only the top beam_width candidates
            # since we are keeping the top n -> tractactable
            next_beam.sort(key = lambda x: x[0])
            beam = next_beam[:beam_width]
    
        return min(path_max for path_max, _ in beam) if beam else 0    
    # simulates one possible future from the current mask state:
     
    # 1. starting from mask, runs up to depth steps (USING THIS AS COMPARED TO A RANDOM NUMBER -> for a bit more determinism)
    # 2. at each step, samples a pool of candidate bits to potentially add
    # 3. greedily picks the best candidate 
    #    -> HOW?: whichever bit, when OR'd in, minimizes the boundary score
    # 4. it tracks the worst (highest) boundary seen across all steps in this rollout
    def rollout_score(mask, depth):
        cur_mask = mask
        worst = 0
    
        for _ in range(depth):
            if cur_mask == all_mask:
                break

            # randomness comes from sampled_candidate_bits 
            # — it draws a random subset of candidates each rollout 
            # - running many roolouts and taking the min approxmates: 
            #    - "whats the best outcome I could reasonably hope for from this state?" 
            #    — useful for guiding a search toward promising branches without exhaustively exploring all paths.
    
            candidate_bits = sampled_candidate_bits(cur_mask, pool_size = lookahead_pool_size,
                                                    sample_size = lookahead_sample_size)
    
            # choose greedily among sampled candidates during rollout
            best_bit = min(candidate_bits, key = lambda bit: boundary_after_mask(cur_mask | bit))
    
            cur_mask |= best_bit
            worst = max(worst, boundary_after_mask(cur_mask))
    
        return worst

    def mc_k_score(mask, depth, num_rollouts = 20):
        if depth == 0 or mask == all_mask:
            return 0
    
        scores = [rollout_score(mask, depth) for _ in range(num_rollouts)]
        return min(scores)  # optimistic: luckiest plausible future
        # OR return sum(scores) / len(scores)  # expected
        # OR return max(scores)  # conservative
        # MAYBE: a COST FUNCTION???

    selected_mask = 0
    internal_order = []
    max_boundary_seen = 0

    pbar = tqdm(total = n, desc = "Greedy MC lookahead", unit = "vertex", dynamic_ncols = True)

    while selected_mask != all_mask:
        best_v = None
        best_bit = None
        best_score = None

        if top_level_pool_size is None:
            candidate_bits = list(iter_bits(all_mask ^ selected_mask))
        else:
            candidate_bits = candidate_bits_by_immediate(selected_mask,  pool_size = top_level_pool_size)

        for bit in candidate_bits:
            v = node_of[bit]
            new_mask = selected_mask | bit

            immediate = boundary_after_mask(new_mask)
            future = best_k_score(new_mask, depth = k - 1)
            # future = beam_k_score(new_mask, depth = k - 1, beam_width = 50,
            #                       branch_pool_size = lookahead_pool_size,
            #                       branch_sample_size = lookahead_sample_size)
            # future = mc_k_score(new_mask, k - 1, num_rollouts = 20)

            score = (max(immediate, future), 
                     immediate,
                     positions[v][0],
                     positions[v][1],
                     str(v))

            if best_score is None or score < best_score:
                best_score = score
                best_v = v
                best_bit = bit

        internal_order.append(best_v)
        selected_mask |= best_bit

        current_boundary = boundary_after_mask(selected_mask)
        max_boundary_seen = max(max_boundary_seen, current_boundary)

        pbar.set_postfix({"chosen": str(best_v), "boundary": current_boundary,
                          "max_seen": max_boundary_seen, "remaining": n - len(internal_order)})
        pbar.update(1)

    pbar.close()

    final_order = [start] + internal_order + [end]

    max_cut, cut, boundaries = compute_cut_fast(G, final_order)

    return final_order, cut, boundaries


def greedy_lookahead(G, positions, start = "I", end = "O", k = 3):
    # k -> number of steps to look ahead
    # INTUITION: choose the next node that seems best not just immediately... but over the next k moves
    remaining = {n for n in G.nodes if n not in (start, end)}
    order = [start]
    selected = {start}

    def best_k_score(selected, remaining, depth):
        # print(selected, remaining, depth)
        #     at each level: try every candidate vertex and recurse,  build up the worst boundary seen along each path
        #     return: the path's minimum worst-case  -> the best we can do from this state if we select greedily for depth moves
        # print(depth)
        # returns the minimum achievable peak boundary over the next depth steps
        # if the depth is 0 or the list of remaining nodes is empty
        # return -> BASE CASE
        if depth == 0 or not remaining:
            return 0

        # recursively look for the best score
        best = None
        # can prob do some dynamic programming here???
        for v in remaining:
            new_selected = selected | {v}
            
            # boundary after adding v at this step (imediate damage of choosing v)
            b = boundary_size(G, new_selected)

            # simulate the future (RECURSVELY)
            # minimum worst-case boundary achievable over the remaining depth-1 steps
            future = best_k_score(new_selected, remaining - {v}, depth - 1)
            
            # the peak boundary over this entire path is whichever is bigger — 
            # the damage right now (b), or future damage
            # cant avoid the bigger of the two -> gonna happen on this path regardless
            worst = max(b, future) # peak boundary over this subpth
            # keep the candidate that gives the lowest peak
            if best is None or worst < best:
                best = worst
        return best

    # gredly extend the ordering one vertex at a time
    while remaining:
        best_v = None
        best_score = None

        for v in remaining:
            new_selected = selected | {v}
            # immediate boundary if we pick v next
            b = boundary_size(G, new_selected)
            # best worst-case boundary over the following k-1 steps
            future_worst = best_k_score(new_selected, remaining - {v}, k - 1)
            # minimise peak boundary over the k-step window
            score = (
                max(b, future_worst), # prim: peak over k step window
                b,                    # sec: prefer lower boundary right now
                positions[v][0],      # horizonal position (prefer lower)
                positions[v][1]       # vertical position (prefer lower)
            )
            if best_score is None or score < best_score:
                best_score = score
                best_v = v

        # best vertex found and move to the next step
        order.append(best_v)
        selected.add(best_v)
        remaining.remove(best_v)

    order.append(end)
    max_cut, boundaries = compute_cut(full_order(order), G)
    
    return order, max_cut, boundaries

def greedy_lookahead_optimized(G, positions, start="I", end="O", k=3):
    # every node that isnt the start or end into a list
    # these are the nodes whose ordering we need to optimise... AND the ones we can permute freelhy
    internal = [v for v in G.nodes if v not in (start, end)]
    # sort by x coordinate, then y coordinate, then the string representation (alphabeticaly)
    # sort before assignment is important: 
    # it makes the bit assignment deterministic across calls
    # this is required for the lru_cache to behave correctly... (I THINK)
    internal.sort(key = lambda v: (positions[v][0], positions[v][1], str(v)))

    # get number of nodes
    n = len(internal)

    # edge case: no intenal nodes
    if n == 0:
        # return order, max_cut, all the cuts and edges between each cut is calcuated
        final_order = [start, end]
        max_cut, cut, boundaries = compute_cut_fast(G, final_order)
        return final_order, max_cut, cut, boundaries

    # ---------------------------------------------------------------------------------------------------
    # BITMAP INDEX CONSTRUCTION
    # ---------------------------------------------------------------------------------------------------
    # builds a lookup from node to its unique powere of 2 bit
    # node at index 0 -> bit 1 (0b001)
    # node at index 1 -> bit 2 (0b010) etc
    # this is to let any subset of nodes be represented as a single integer
    bit_of = {v: 1 << i for i, v in enumerate(internal)}
    # reverse map: given a power of two bit, what node does it represent
    # used later to recoved which ndoe was just added
    node_of = {1 << i: v for i, v in enumerate(internal)}
    # a bitmask with all n bits set -> represents the satet where every internal node has been placed
    # ex. n=3: (1<<3)-1 = 7 = 0b111 (used as the termination condition in the lookahead)
    all_mask = (1 << n) - 1
    # ---------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------
    # EDGE WEIGHT PREO
    # ---------------------------------------------------------------------------------------------------
    edge_to_start = {v: 0 for v in internal}
    edge_to_end = {v: 0 for v in internal}
    internal_adj = {v: {} for v in internal}
    # ---------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------------------------------------------
    def add_internal_edge(a, b):
        # increment the multiplicity ocunter between two indernal nodes
        # called twice for each internal edge (onnce for each direction) -> adjecency is symmetric
        internal_adj[a][b] = internal_adj[a].get(b, 0) + 1

    # walk every edge once and classifies it in the multigraph
    for u, v, _ in G.edges(keys = True):
        # self-loops are skipped
        if u == v:
            continue
        # internal-internal edges update the internal adj function above (IN BOTH DIREACTIONS)
        if u in bit_of and v in bit_of:
            add_internal_edge(u, v)
            add_internal_edge(v, u)
        # internal-start or start-internal edges incremente edge_to_start
        elif u in bit_of and v == start:
            edge_to_start[u] += 1
        elif v in bit_of and u == start:
            edge_to_start[v] += 1
        # internal-end or end-internal efges incremente edge_to_start
        elif u in bit_of and v == end:
            edge_to_end[u] += 1
        elif v in bit_of and u == end:
            edge_to_end[v] += 1

    # generator function
    # yields each individual set bit of mask one at a time
    def iter_bits(mask):
        while mask:
            # in two's compleent -mask flips all bits and adds one
            # this isolates exatly the lowers set bit
            # this helper is used in best_k_score underneath to enumerate tevery remaining unplaced node
            bit = mask & -mask # all bits become 0 except the rightmost 1 bit
            # EXAMPLE
            # mask = 0b01011000
            # -mask = 0b10100111 + 0b0000001 = 0b10101000
            # mask & -mask = 0b01011000 AND 0b10101000 = 0b00001000
            yield bit
            # XORing it out clears that bit for the next iteration
            # loop ends when 0b000...0
            mask ^= bit

    # memoises this function by its integer argument: 
    # -> because many different greedy paths can arrive at the same set of placed nodes
    # -> results are shared rather than recomputed
    @lru_cache(maxsize = None)
    def boundary_after_mask(mask):
        # base case: no node placed yet
        # 
        if mask == 0:
            count = 0
            # for the set of vertices
            for u, v, _ in G.edges(keys = True):
                # ignore selg loops
                if u == v:
                    continue
                # true only when one node is enpoint
                if (u == start) != (v == start):
                    # counts edges leaving start toward internal nodes and internal toward the end node
                    count += 1
            return count

        # get which bit (NODE) was added mnost recently -> lowest set bit as the last addition.
        added_bit = mask & -mask
        # the canonical prev state is always mask with that bit cleared
        prev_mask = mask ^ added_bit

        # looks up which node the bit represnts
        v = node_of[added_bit]
        # recursively fetches the boundary for the previous state (already cached form earlier call?)
        # O(n)?
        old_boundary = boundary_after_mask(prev_mask)

        # when a node c is added to the placed set
        # its edges to the start node are now internal (both endpoints placed)
        edges_to_selected = edge_to_start[v]
        edges_to_unselected = edge_to_end[v]

        # for each internal neighbor
        for nbr, multiplicity in internal_adj[v].items():
            nbr_bit = bit_of[nbr]

            # edges already placed (to be subtracted)
            if prev_mask & nbr_bit:
                edges_to_selected += multiplicity
            # edges not yer placed (to be added)
            else:
                edges_to_unselected += multiplicity

        # add and subtract from old boundary
        return old_boundary - edges_to_selected + edges_to_unselected

    # cache key is (mask, depth)
    @lru_cache(maxsize = None)
    def best_k_score(mask, depth):
        # base cases
        if depth == 0 or mask == all_mask:
            return 0

        # get the remaining bitmask of the nodes not yer placed 
        # XOR all against the all_mask
        # this flips every paced bit to 0 and every unplaced bit to 1
        # best will track the minimum worst-case score found across all candidate next nodes
        remaining_mask = all_mask ^ mask
        best = None

        # lookahead loop
        # for each unplaced node (one per bit)
        for bit in iter_bits(remaining_mask):
            # 1. hypothetically pace it
            new_mask = mask | bit
            # 2. compute the immediate boundary cost of that placement
            immediate = boundary_after_mask(new_mask)
            # recursively ask what the best WORST case achiavalbe in the next depth-1 steps
            future = best_k_score(new_mask, depth - 1)

            # worst is the max obtained
            worst = max(immediate, future)

            # track the candidate that minimises this peak
            # minimax search: we choose the move that minimises the orst-case future exposure
            if best is None or worst < best:
                best = worst

        return best
    # ---------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------
    # ACTUAL COMPUTATION
    # ---------------------------------------------------------------------------------------------------
    # bitmask tracking which internal nodes have been placed so far
    # starts at 0 (no bits set = nobody placed yet) and grows by ORing in one bit per iteration until it equals all_mask
    selected_mask = 0
    # output list
    internal_order = []
    # running max of the boundary cut across all placement steps
    max_boundary_seen = 0

    # tqdm progress bar
    pbar = tqdm(total = n, desc = "Greedy lookahead", unit = "vertex", dynamic_ncols = True)

    # keep looping until every internal node has been placed
    while selected_mask != all_mask:
        # get the remainign nodes (mask of bits that are not yey placed -> candidate nodes)
        # XOR flips every bit
        # bits that are set in selected_mask become 0 and bits that are 0 become 1
        remaining_mask = all_mask ^ selected_mask

        # best running vars
        best_v = None
        best_bit = None
        best_score = None

        # for each node that has not yet been placed
        for bit in iter_bits(remaining_mask):
            # get the node object fromt he precomputed reverse map
            v = node_of[bit]

            # new mask -> hypotehtically place this candidate 
            # the OR its bits into the current placement mask
            # just test what happens
            new_mask = selected_mask | bit
            # compute the immetidate boundary cost
            immediate = boundary_after_mask(new_mask)
            # ask the lookahead: if i make this move what is the minimum achievable worst-case boundary over the next steps k-1
            # SAME AS ABOVE
            future = best_k_score(new_mask, k - 1)

            score = (
                max(immediate, future), # primary: minimise the peak boundary over the next k steps
                immediate,              # secondary: if all are the same, prioritise the current immediate minimum
                positions[v][0],        # otherwise: x pos -> y pos -> alphabetical name
                positions[v][1],       
                str(v)
            )

            # update best parameter
            if best_score is None or score < best_score:
                best_score = score
                best_v = v
                best_bit = bit

        # append the best node
        internal_order.append(best_v)
        # in-place bitwise or with current selected maks to update it
        selected_mask |= best_bit
        # updte current bounradu and max boundary
        current_boundary = boundary_after_mask(selected_mask)
        max_boundary_seen = max(max_boundary_seen, current_boundary)

        # udpate progress bar
        pbar.set_postfix({
            "chosen": str(best_v),
            "boundary": current_boundary,
            "max_seen": max_boundary_seen,
            "remaining": n - len(internal_order)})
        pbar.update(1)
    pbar.close()

    # get final order, max_cut and cut
    final_order = [start] + internal_order + [end]

    max_cut, cut, boundaries = compute_cut_fast(G, final_order)

    return final_order, cut, boundaries
