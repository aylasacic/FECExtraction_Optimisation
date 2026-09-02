import pyzx as zx
import random
import fractions
import math
import networkx as nx
import numpy as np
from collections import defaultdict

PHASES = [fractions.Fraction(0, 1), fractions.Fraction(1, 2), fractions.Fraction(1, 1), fractions.Fraction(3, 2)]

def random_zx_graph(n_in = 4, n_out = 4, n_spiders = 12, edge_p = 0.4, seed = None):
    # for plotting later on -> helps keep output and input quite seperated from the diagram
    max_row = max(6, int(math.sqrt(n_spiders) * 5))

    # random stuff
    rng = random.Random(seed)

    # initialise graph
    g = zx.Graph()

    # input = output qbits 
    # although rewriting can add ancilae, it also adds post-selection
    # therefore, the initial input and outputs should always be the same (if unitary)
    inputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit = i, row = 0) for i in range(n_in)]
    outputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit = i, row = max_row) for i in range(n_out)]

    # ------------------------------------------------------------------------------
    # generate random number of spiders
    spiders = []
    for _ in range(n_spiders):
        # choose X or Z
        vtype = rng.choice([zx.VertexType.Z, zx.VertexType.X])
        # random phase (CURRENTLY ONLY 0 PHASE -> CHANGE LATER)
        phase = rng.choice(PHASES)
        # add the gate (spider) at a random position in the diagram
        # to a random quibit
        v = g.add_vertex(vtype, qubit = rng.randint(0, max(n_in, n_out) + 3)+random.random()*2,
                         row = rng.randint(1, max_row - 1)+random.random()*2,
                         phase = phase)
        spiders.append(v)
    # ------------------------------------------------------------------------------

    # get the degree of the spider
    def get_deg(v):
        return g.vertex_degree(v)

    # see if the edge is valid (3-ary graph)
    def can_connect(v):
        return get_deg(v) < 3

    def hopf(v1, v2):
        pass
        # for edge in v1

    # ------------------------------------------------------------------------------
    def add_edge_if_possible(v1, v2, A):
        # if the spiders are not already of degree 3
        # and we are not looking at the same vertices
        # self-loops an be disregarded in later ZX rewriting, but we can do it here 
        if can_connect(v1) and can_connect(v2) and v1 != v2:
            # add a simple edge
            # since phases are removed, I also removed hadamard gates for now
            # ADD HADAMARD EDGES LATR
            g.add_edge(g.edge(v1, v2), edgetype = zx.EdgeType.SIMPLE)
            A[v1, v2] = 1
            A[v2, v1] = 1
        return A
    # ------------------------------------------------------------------------------

    adjc = np.zeros((n_spiders+n_in+n_out, n_spiders+n_in+n_out))
    # print(adjc)
    # ------------------------------------------------------------------------------
    # 1. INPUTS and OUTPUTS
    # go over the input and outputs
    for b in inputs + outputs:
        # get all non-3-deg spiders
        # all should be valid in the beginning
        candidates = [v for v in spiders if can_connect(v)]
        # if none are possible, error? 
        # this happens later on in generation but I am not sure how to go about it
        # either error or just pass
        if not candidates:
            raise ValueError("")
            # continue
        # choose a random qbit to connect to 
        v = rng.choice(candidates)
        # add the edge input/output -> random spider
        g.add_edge(g.edge(b, v), edgetype = zx.EdgeType.SIMPLE)
        adjc[b, v] = 1
        adjc[v, b] = 1
        
    # ------------------------------------------------------------------------------
    # print(adjc)
    
    # 2. INNER GRAPH
    for i in range(len(spiders)):
        for j in range(i + 1, len(spiders)):
            # randomyl add an edge
            # between two spiders
            if rng.random() < edge_p:
                adjc = add_edge_if_possible(spiders[i], spiders[j], adjc)

    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))
    return g, adjc