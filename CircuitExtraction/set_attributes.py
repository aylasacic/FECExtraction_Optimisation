import pyzx as zx
import random
import fractions
import numpy as np
from collections import defaultdict, deque
from pyzx.utils import VertexType
from pyzx.circuit.gates import TargetMapper, CNOT

_PATCH_SENTINEL = "_targetmapper_patched_once"

'''
this code is used to initialise some new attraibutes of TargetMapper in PyZX package.
It mainly relates to qubit allocation after freeing which withouht this code happens at a timestep
right after the qubit was freed even if the new qubit is initialised some k timesteps after.
this code fixes it so that qubits are allocated a timestep before their use (ancilae qubits)
'''
# ----------------------------------------------------------------------------------------------------------------
# OLD
# ----------------------------------------------------------------------------------------------------------------

# def apply_targetmapper_patches():
#     if getattr(TargetMapper, _PATCH_SENTINEL, False):
#         return

#     if not hasattr(TargetMapper, "_pristine"):
#         TargetMapper._pristine = {"remove_label": TargetMapper.remove_label, "next_row": TargetMapper.next_row, 
#                                   "next_row_or_default": TargetMapper.next_row_or_default}

#     orig_remove_label = TargetMapper._pristine["remove_label"]
#     orig_next_row = TargetMapper._pristine["next_row"]

#     def live_frontier(self):
#         live = [self._rows[k] for k in self._labels if k in self._rows]
#         return max(live) if live else self.max_row()

#     def remove_label(self, l):
#         orig_remove_label(self, l)
#         self._rows.pop(l, None)

#     def next_row_or_default(self, l, default):
#         if l not in self._labels:
#             return live_frontier(self) - 1
#         return self._rows.get(l, live_frontier(self) - 1)

#     def next_row(self, l):
#         if l not in self._rows:
#             return max(live_frontier(self) - 1, 0)
#         return orig_next_row(self, l)

#     TargetMapper.remove_label = remove_label
#     TargetMapper.next_row_or_default = next_row_or_default
#     TargetMapper.next_row = next_row

#     setattr(TargetMapper, _PATCH_SENTINEL, True)
# ----------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# NEW
# ----------------------------------------------------------------------------------------------------------------
def _cnot_to_graph_endpoints_only(self, g, q_mapper, _c_mapper):
    r = max(q_mapper.next_row(self.target), q_mapper.next_row(self.control))
    t = self.graph_add_node(g, q_mapper, VertexType.X, self.target, r)
    c = self.graph_add_node(g, q_mapper, VertexType.Z, self.control, r)
    g.add_edge((t, c))
    q_mapper.set_next_row(self.target, r + 1)
    q_mapper.set_next_row(self.control, r + 1)
    g.scalar.add_power(1)

def apply_targetmapper_patches():
    CNOT.to_graph = _cnot_to_graph_endpoints_only
    
    if getattr(TargetMapper, _PATCH_SENTINEL, False):
        return

    if not hasattr(TargetMapper, "_pristine"):
        TargetMapper._pristine = {"remove_label": TargetMapper.remove_label, "next_row": TargetMapper.next_row, 
                                  "next_row_or_default": TargetMapper.next_row_or_default}

    orig_remove_label = TargetMapper._pristine["remove_label"]
    orig_next_row = TargetMapper._pristine["next_row"]

    def live_frontier(self):
        live = [self._rows[k] for k in self._labels if k in self._rows]
        return max(live) if live else self.max_row()

    def remove_label(self, l):
        orig_remove_label(self, l)
        self._rows.pop(l, None)

    def next_row_or_default(self, l, default):
        if l not in self._labels:
            return live_frontier(self) - 1
        return self._rows.get(l, live_frontier(self) - 1)

    def next_row(self, l):
        if l not in self._rows:
            return max(live_frontier(self) - 1, 0)
        return orig_next_row(self, l)

    TargetMapper.remove_label = remove_label
    TargetMapper.next_row_or_default = next_row_or_default
    TargetMapper.next_row = next_row

    setattr(TargetMapper, _PATCH_SENTINEL, True)

def repack_gate(g):
    rows   = {v: g.row(v) for v in g.vertices()}
    inputs = set(g.inputs())

    # union the two endpoints of every same-row edge (i.e. each 2-qubit gate)
    parent = {v: v for v in g.vertices()}
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    for e in g.edges():
        s, t = g.edge_st(e)
        if rows[s] == rows[t]:
            parent[find(s)] = find(t)

    groups = defaultdict(list)
    for v in g.vertices():
        groups[find(v)].append(v)

    # schedule each gate-group as ONE column, left to right (ASAP)
    order  = sorted(groups.values(), key=lambda grp: min(rows[v] for v in grp))
    depth  = defaultdict(int)            # next free column per qubit position
    newrow = {}
    for grp in order:
        pos = [g.qubit(v) for v in grp]
        r = max(depth[p] for p in pos)
        for v in grp: newrow[v] = r
        for p in pos: depth[p] = r + 1

    # pull each InitAncilla (a source group) up to just before its first use
    for grp in order:
        if any(v in inputs for v in grp):
            continue
        if all(not any(rows[u] < rows[v] for u in g.neighbors(v)) for v in grp):
            succ = [newrow[u] for v in grp for u in g.neighbors(v) if rows[u] > rows[v]]
            if succ:
                for v in grp:
                    newrow[v] = min(succ) - 1

    for v in g.vertices():
        g.set_row(v, newrow[v])

# ----------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------