import random
import pyzx as zx
import numpy as np
import fractions
import math
import networkx as nx
import networkx as nx
from pyzx.utils import EdgeType, VertexType

_VT_NAMES = _int_constants(VertexType)
_ET_NAMES = _int_constants(EdgeType)

def print_graph(g):
    print("\nVertices")
    for v in g.vertices():    
        print(f"{v}: " f"{g.type(v).name}" f"(phase = {g.phase(v)}, " f"row = {g.row(v)}, "f"q = {g.qubit(v)})")

    print("\nAdjacency")
    for v in g.vertices():
        nbrs = ", ".join(map(str, g.neighbors(v)))
        print(f"{v} -> [{nbrs}]")

    print("\nEdges")
    seen = set()

    for v in g.vertices():
        for u in g.neighbors(v):
            edge = tuple(sorted((v, u)))
            if edge in seen:
                continue
            seen.add(edge)
            etype = g.edge_type(g.edge(v, u)).name

            print(f"{v} -- {u} ({etype})")
            
def _int_constants(cls):
    out: dict[int, str] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        val = getattr(cls, name)
        if isinstance(val, int) and not isinstance(val, bool):
            out.setdefault(int(val), name)
    return out


def vtype_name(t):
    if hasattr(t, "name"):
        return t.name
    return _VT_NAMES.get(int(t), f"UNKNOWN_VERTEX_{int(t)}")


def etype_name(t):
    if hasattr(t, "name"):
        return t.name
    return _ET_NAMES.get(int(t), f"UNKNOWN_EDGE_{int(t)}")


def nx_graph(g):
    nxg = nx.MultiGraph()
    pos = {}
    # authoritative wire index from pyzx's ordered boundary lists
    in_index = {v: i for i, v in enumerate(g.inputs())}
    out_index = {v: i for i, v in enumerate(g.outputs())}

    for v in g.vertices():
        vt = vtype_name(g.type(v))
        if vt == "BOUNDARY":
            if v in in_index:
                key, bidx = f"BI_{v}", in_index[v]
            elif v in out_index:
                key, bidx = f"BO_{v}", out_index[v]
            else:
                # a boundary in neither list: possible after hand-editing a diagram
                key, bidx = f"B_{v}", None
        else:
            key, bidx = f"I_{v}", None
        nxg.add_node(v, key = key, vtype = vt, vtype_int = int(g.type(v)),
                     phase = g.phase(v), row = g.row(v), qubit = g.qubit(v),
                     boundary_index = bidx)
        pos[v] = (g.row(v), -g.qubit(v))

    for e in g.edges():
        s, t = g.edge_st(e)
        nxg.add_edge(s, t, edge_type = etype_name(g.edge_type(e)),
                     edge_type_int = int(g.edge_type(e)))

    nx.set_node_attributes(nxg, pos, "pos")
    return nxg, pos

def h_boxes_to_edges(g, copy: bool = True, boundary_safe: bool = True):
    if copy:
        g = g.copy()
    skipped = []
    for v in list(g.vertices()):
        if int(g.type(v)) != int(VertexType.H_BOX):
            continue
        nbs = list(g.neighbors(v))
        if g.vertex_degree(v) != 2 or len(nbs) != 2:
            skipped.append(v)
            continue
        a, b = nbs
        hads = 1 + sum(int(g.edge_type(g.edge(v, u))) == int(EdgeType.HADAMARD)
                       for u in (a, b))
        new_type = EdgeType.HADAMARD if hads % 2 else EdgeType.SIMPLE
        if g.connected(a, b):
            skipped.append(v) 
            continue
        g.remove_vertex(v)
        if (new_type == EdgeType.HADAMARD and boundary_safe
                and (int(g.type(a)) == int(VertexType.BOUNDARY)
                     or int(g.type(b)) == int(VertexType.BOUNDARY))):
            mid = g.add_vertex(VertexType.Z, qubit=g.qubit(a), row=g.row(a), phase=0)
            g.add_edge((a, mid), EdgeType.HADAMARD)
            g.add_edge((mid, b), EdgeType.SIMPLE)
        else:
            g.add_edge((a, b), new_type)
    return g, skipped

def collapse_io(nxg, collapse_horizontally = False, collapse_vertically = False):
    
    collapsed_nxg = nx.MultiGraph(nxg) 
    horizontal_collapsed_nxg = nx.MultiGraph()

    # get all i/o nides
    input_nodes = [v for v in collapsed_nxg.nodes if collapsed_nxg.nodes[v]["key"].startswith("BI_")]
    output_nodes = [v for v in collapsed_nxg.nodes if collapsed_nxg.nodes[v]["key"].startswith("BO_")]
    internal_nodes = [v for v in collapsed_nxg.nodes if collapsed_nxg.nodes[v]["key"].startswith("I")]

    # print(internal_nodes)

    # save the boundary attributes for later (I think this can be done wi the adjc matrix?)
    # KEEP FOR NOW
    boundary_attrs = {v: dict(collapsed_nxg.nodes[v]) for v in input_nodes + output_nodes}
    boundary_links = {v: list(collapsed_nxg.neighbors(v)) for v in input_nodes + output_nodes}

    IN = "I"
    OUT = "O"

    # all positions
    pos = nx.get_node_attributes(collapsed_nxg, "pos")

    # new i/o positin = mean of all i/o positions
    in_pos = tuple(np.mean([pos[b] for b in input_nodes], axis = 0))
    out_pos = tuple(np.mean([pos[b] for b in output_nodes], axis = 0))

    # new i/o node (collapse as in viviennes paper)
    collapsed_nxg.add_node(IN, pos = in_pos, key = "IN")
    collapsed_nxg.add_node(OUT, pos = out_pos, key = "OUT")
    
    pos[IN] = in_pos
    pos[OUT] = out_pos

    if collapse_horizontally == True:
        for node in pos:
            pos[node] = (pos[node][0], in_pos[1])
            
    # reconnect the graph as before
    for b in input_nodes:
        q = nxg.nodes[b]["boundary_index"]
        for nbr in list(collapsed_nxg.neighbors(b)):
            for k, data in collapsed_nxg.get_edge_data(b, nbr).items():
                collapsed_nxg.add_edge(IN, nbr, edge_type = data.get("edge_type", "SIMPLE"), boundary_qubit = q, boundary_kind = "I")
        
    for b in output_nodes:
        q = nxg.nodes[b]["boundary_index"]
        for nbr in list(collapsed_nxg.neighbors(b)):
            collapsed_nxg.add_edge(OUT, nbr, edge_type = "SIMPLE", boundary_qubit = q, boundary_kind = "O")

    # remove old input and output nodes
    collapsed_nxg.remove_nodes_from(input_nodes + output_nodes)

    # JUST FOR PLOTTING
    colour_map = []
    for node in collapsed_nxg.nodes:
        if collapsed_nxg.nodes[node]["key"].startswith("IN") or collapsed_nxg.nodes[node]["key"].startswith("OUT"):
            # colour_map.append('#E87C45')
            colour_map.append('#ffffff')
        else: 
            # colour_map.append('skyblue') 
            colour_map.append('#000000') 

    return collapsed_nxg, pos, boundary_attrs, boundary_links, colour_map

def split_vertex(G, boundary_nodes = ("I", "O")):
    H = nx.MultiGraph(G)
    internal_nodes = [v for v in list(H.nodes) if v not in boundary_nodes]

    leg_parent = {}

    for v in internal_nodes:
        # group incident multiedges by neighbour
        neighbour_edges = {}

        for u, w, k, data in list(H.edges(v, keys = True, data = True)):
            nbr = w if u == v else u

            if nbr == v:
                # ignore self-loops for now
                continue

            neighbour_edges.setdefault(nbr, []).append((u, w, k, dict(data)))

        for nbr, edges in neighbour_edges.items():
            if len(edges) <= 1:
                continue

            for idx, (u, w, k, data) in enumerate(edges):
                if not H.has_edge(u, w, key = k):
                    continue

                H.remove_edge(u, w, key = k)

                leg = ("leg", v, nbr, k, idx)
                # leg = ("leg", v, nbr)

                v_attrs = dict(H.nodes[v])
                leg_attrs = dict(v_attrs)
                leg_attrs["phase"] = fractions.Fraction(0, 1)
                leg_attrs["key"] = f"LEG_{v}_{nbr}_{k}_{idx}"
                leg_attrs["is_leg_spider"] = True
                leg_attrs["parent"] = v

                if "pos" in v_attrs:
                    x, y = v_attrs["pos"]
                    leg_attrs["pos"] = (x - 0.05 * (idx + 1), y)

                H.add_node(leg, **leg_attrs)

                outside_edge_data = dict(data)
                outside_edge_data.setdefault("edge_type", "SIMPLE")
                
                H.add_edge(nbr, leg, **outside_edge_data)
    
                leg_parent_edge_data = {"edge_type": "SIMPLE", "is_leg_parent_edge": True}

                if "boundary_kind" in data:
                    leg_parent_edge_data["boundary_kind"] = data.get("boundary_kind")
                    leg_parent_edge_data["boundary_qubit"] = data.get("boundary_qubit")
                    leg_parent_edge_data["boundary_key"] = data.get("boundary_key")
                    leg_parent_edge_data["original_boundary"] = data.get("original_boundary")
                
                H.add_edge(leg, v, **leg_parent_edge_data)

                leg_parent[leg] = v

    return H, leg_parent


