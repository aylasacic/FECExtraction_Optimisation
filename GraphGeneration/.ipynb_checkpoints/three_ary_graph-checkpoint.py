import pyzx as zx
import random
import fractions
import math
import numpy as np

# PHASES_NONE = [fractions.Fraction(0, 1)]
PHASES = [fractions.Fraction(0, 1), fractions.Fraction(1, 2), fractions.Fraction(1, 1), fractions.Fraction(3, 2)]

def random_3ary_zx_graph(n_in = 4, n_out = 4, n_spiders = 12, seed = None, include_phases = False, 
                         circuit_like = False, circuit_layers = None, layout_jitter = 0.12):
    # rng = random.Random(seed)

    # if n_spiders < 4:
    #     raise ValueError("> 4 spiders needed (NOT AN ISSUE IF RANDOM IS SET CORRECTLY)")

    # n_boundaries = n_in + n_out

    # # each boundary consumes one spider degree slot
    # # so 3 * # spiders = # edges required
    # # MINUS the required boundary connection
    # total_remaining_stubs = 3 * n_spiders - n_boundaries

    # if total_remaining_stubs < 0:
    #     raise ValueError("more boundaries than avialable spiders")

    # if total_remaining_stubs % 2:
    #     raise ValueError("odd # stubs")

    # # output row
    # max_row = max(6, int(math.sqrt(n_spiders) * 5))
    # g = zx.Graph()

    # # create intput and output spiders (BOUNDARIES)
    # inputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit = i, row = 0) for i in range(n_in)]

    # outputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit = i, row = max_row) for i in range(n_out)]

    # boundaries = inputs + outputs
    # spiders = []

    # # add the intenral spiders
    # for _ in range(n_spiders):

    #     v = g.add_vertex(rng.choice([zx.VertexType.Z, zx.VertexType.X]),
    #                      qubit = rng.uniform(0, max(n_in, n_out) + 3),
    #                      row = rng.uniform(1, max_row - 1),
    #                      phase = rng.choice(PHASES) if include_phases == True else None
    #     )
    #     spiders.append(v)

    # # dictionaty that will count number of bondaries for each internal spiders
    # boundary_count = {v: 0 for v in spiders}

    # # for each boundary
    # for b in boundaries:
    #     # get candidate spiders tha can be connected top it
    #     candidates = [v for v in spiders if boundary_count[v] < 3]

    #     # if there are no candidates (fail)
    #     if not candidates:
    #         raise RuntimeError("failed 1")

    #     # select a random candidate
    #     v = rng.choice(candidates)
    #     # and connect it to the current selected spider
    #     g.add_edge(g.edge(b, v), edgetype = zx.EdgeType.SIMPLE,)
    #     # increment the boundary count for that spider
    #     boundary_count[v] += 1

    # # get the residual degrees 
    # residual_deg = {}

    # # for each spider
    # for v in spiders:
    #     # resodual is 3 (since we want 3-ary) - the number of current edges going out from it
    #     residual_deg[v] = 3 - boundary_count[v]

    # # try this many times to connect the remainign
    # max_attempts = 1000
    # success = False

    # for _ in range(max_attempts):

    #     stubs = []

    #     # for the vertex and the number of remaining degrees it has
    #     for v, deg in residual_deg.items():
    #         # build a list by repeating each vertex v according to its degree deg
    #         stubs.extend([v] * deg)

    #     # shuffle the list
    #     rng.shuffle(stubs)

    #     edges = set()
    #     valid = True

    #     # while the list is not empty
    #     while stubs:
    #         # pop out two stubs
    #         a = stubs.pop()
    #         b = stubs.pop()

    #         # if they're the same restart
    #         if a == b:
    #             valid = False
    #             break

    #         e = tuple(sorted((a, b)))

    #         # if the edge is already present, restart (dont add double edges?)
    #         if e in edges:
    #             valid = False
    #             break

    #         # otherwise add the edge
    #         edges.add(e)

    #     # if valid is still true (SHOULD BE)
    #     if valid:
    #         # restart for next
    #         success = True
    #         spider_edges = list(edges)
    #         break

    # # if we failed, graph cannot be created
    # if not success:
    #     raise RuntimeError("failed 2 (stubs)")

    # # for the set of vertices found
    # for u, v in spider_edges:
    #     # add edges between them
    #     g.add_edge(g.edge(u, v), edgetype = zx.EdgeType.SIMPLE,)

    # # for each spider
    # for v in spiders:
    #     # get degree adn if its less than 3 -> fail
    #     deg = g.vertex_degree(v)
    #     if deg != 3:
    #         raise RuntimeError(f"{v} degree is {deg}")

    # # get the vertices
    # verts = list(g.vertices())
    # # enimerate them
    # idx = {v: i for i, v in enumerate(verts)}
    # # ADJECENCY MTRIX
    # # create an empty n by n matrix where n is the # of vertices
    # A = np.zeros((len(verts), len(verts)), dtype = int)

    # # fill up the matrix 
    # # i added this before when I thought that I would use it still
    # # might be useful later
    # for e in g.edges():
    #     s, t = g.edge_st(e)
    #     A[idx[s], idx[t]] = 1
    #     A[idx[t], idx[s]] = 1

    # # set the inputs and outpus into the grpah at end
    # g.set_inputs(tuple(inputs))
    # g.set_outputs(tuple(outputs))

    # return g, A
    rng = random.Random(seed)

    if n_spiders < 4:
        raise ValueError("> 4 spiders needed")

    if layout_jitter < 0:
        raise ValueError("layout_jitter must be non-negative")

    n_boundaries = n_in + n_out
    total_remaining_stubs = 3 * n_spiders - n_boundaries

    if total_remaining_stubs < 0:
        raise ValueError("more boundaries than available spider degree slots")

    if total_remaining_stubs % 2:
        raise ValueError("odd number of remaining stubs")

    n_wires = max(n_in, n_out, 1)
    max_row = max(6, int(math.sqrt(n_spiders) * 5))

    g = zx.Graph()

    inputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit = i, row = 0) for i in range(n_in)]
    outputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit = i, row = max_row) for i in range(n_out)]

    boundaries = inputs + outputs
    spiders = []

    circuit_positions = None

    if circuit_like:
        if circuit_layers is None:
            # enough grid slots to hold all spiders
            circuit_layers = max(2, math.ceil(n_spiders / n_wires))

        if circuit_layers < 1:
            raise ValueError("circuit_layers must be at least 1")

        # keep internal vertices away from the boundary rows.
        layer_rows = np.linspace(1, max_row - 1, circuit_layers)

        # create one possible spider position per wire/layer pair
        circuit_positions = [(float(qubit), float(row)) for row in layer_rows for qubit in range(n_wires)]

        # in case the requested layer count is too small
        if len(circuit_positions) < n_spiders:
            raise ValueError("Not enough circuit grid positions. Increase circuit_layers.")

        # randomize which grid positions are occupied
        rng.shuffle(circuit_positions)

    for spider_index in range(n_spiders):
        if circuit_like:
            base_qubit, base_row = circuit_positions[spider_index]

            # add a small amount of randomness while preserving the grid
            qubit = base_qubit + rng.uniform(-layout_jitter, layout_jitter)
            row = base_row + rng.uniform(-layout_jitter, layout_jitter)

            # keep coordinates inside the intended drawing region
            qubit = min(max(qubit, 0), n_wires - 1)
            row = min(max(row, 1), max_row - 1)

        else:
            # original unrestricted random placement
            qubit = rng.uniform(0, n_wires + 3)
            row = rng.uniform(1, max_row - 1)

        v = g.add_vertex(rng.choice([zx.VertexType.Z, zx.VertexType.X]), qubit = qubit, 
                         row = row, phase=rng.choice(PHASES) if include_phases else None)
        spiders.append(v)

    # count how many boundaries are attached to each spider
    boundary_count = {v: 0 for v in spiders}

    for b in boundaries:
        candidates = [v for v in spiders if boundary_count[v] < 3]

        if not candidates:
            raise RuntimeError("failed to attach boundaries")

        v = rng.choice(candidates)

        g.add_edge(g.edge(b, v), edgetype = zx.EdgeType.SIMPLE)
        boundary_count[v] += 1

    residual_deg = {v: 3 - boundary_count[v] for v in spiders}

    max_attempts = 10000
    success = False

    for _ in range(max_attempts):
        stubs = []

        for v, degree in residual_deg.items():
            stubs.extend([v] * degree)

        rng.shuffle(stubs)

        edges = set()
        valid = True

        while stubs:
            a = stubs.pop()
            b = stubs.pop()

            if a == b:
                valid = False
                break

            edge = tuple(sorted((a, b)))

            if edge in edges:
                valid = False
                break

            edges.add(edge)

        if valid:
            success = True
            spider_edges = list(edges)
            break

    if not success:
        raise RuntimeError("failed to pair spider stubs")

    for u, v in spider_edges:
        g.add_edge(g.edge(u, v), edgetype=zx.EdgeType.SIMPLE)

    for v in spiders:
        degree = g.vertex_degree(v)

        if degree != 3:
            raise RuntimeError(f"Spider {v} has degree {degree}, expected 3")

    vertices = list(g.vertices())
    index = {v: i for i, v in enumerate(vertices)}
    adjacency = np.zeros((len(vertices), len(vertices)), dtype = int)

    for edge in g.edges():
        source, target = g.edge_st(edge)
        adjacency[index[source], index[target]] = 1
        adjacency[index[target], index[source]] = 1

    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))

    return g, adjacency
