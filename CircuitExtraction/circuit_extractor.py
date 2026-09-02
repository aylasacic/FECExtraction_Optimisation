# ----------------------------------------------------------------------------------------------------------------
# NEW
# ----------------------------------------------------------------------------------------------------------------

import pyzx as zx
import random
import fractions
import numpy as np
from collections import defaultdict, deque
from pyzx.circuit.gates import CNOT, TargetMapper
from pyzx.utils import VertexType
# from set_attributes import apply_targetmapper_patches, _cnot_to_graph_endpoints_only, repack_gate
# from set_attributes import apply_targetmapper_patches, repack_gate

from CircuitExtraction.set_attributes import apply_targetmapper_patches, _cnot_to_graph_endpoints_only, repack_gate

# apply_targetmapper_patches()

PHASES = [fractions.Fraction(0, 1), fractions.Fraction(1, 2), fractions.Fraction(1, 1), fractions.Fraction(3, 2)]

class QubitAllocator:
    def __init__(self, initial_qubits):
        self.next_label = initial_qubits
        self.in_use = set(range(initial_qubits))
        # FIFO
        self.free_labels = deque()          
        self.deferred_labels = []
        # freed this step, not yet quarantined
        self.pending_labels = []           

    def free(self, label):
        if label not in self.in_use:
            raise ValueError(f"unused qubit label {label} cannot be freed")
        self.in_use.remove(label)
        self.pending_labels.append(label)

    def next_step(self):
        # step N-1 frees become reusable
        self.free_labels.extend(self.deferred_labels) 
        # step N frees wait one more
        self.deferred_labels = self.pending_labels     
        self.pending_labels = []

    def allocate(self):
        if self.free_labels:
            label = self.free_labels.popleft()
        else:
            label = self.next_label
            self.next_label += 1
        self.in_use.add(label)
        return label

    def allocate_fresh(self):
        label = self.next_label; self.next_label += 1
        self.in_use.add(label)
        return label

    def reuse(self, label):
        self.free_labels.remove(label)
        self.in_use.add(label)

    def is_active(self, label):
        return label in self.in_use

    def state(self):
        return {"next_label": self.next_label, "in_use": sorted(self.in_use),
                "free_labels": list(self.free_labels), "deferred_labels": list(self.deferred_labels),
                "pending_labels": list(self.pending_labels)}

class CircuitExtractor:
    def __init__(self, num_qubits):
        self.circ = zx.Circuit(num_qubits)
        self.allocator = QubitAllocator(num_qubits)
        self.history = []
        # persistent; replayed on every build_graph/draw
        self.stretches = []            
        self.live_edge_to_qubit = {}
        self.depth = defaultdict(int)
        for i in range(num_qubits):
            # input occupies col 0, wire ready at col 1
            self.depth[i] = 1              

    def add_gate(self, name, *qubits, **kwargs):
        ints = [q for q in qubits if isinstance(q, int)]
        for q in ints:
            if not self.allocator.is_active(q):
                raise ValueError(f"qubit {q} is not active")
        self.circ.add_gate(name, *qubits, **kwargs)
        if ints:                           # same ASAP rule the repack uses
            c = max(self.depth[q] for q in ints)
            for q in ints:
                self.depth[q] = c + 1

    def allocate(self):
        return self.allocator.allocate()

    def _allocate_before(self, deadline):
        # reuse a freed wire only if it goes idle strictly before `deadline`
        # (depth[l] < deadline  =>  its re-init lands early enough to add no column)
        eligible = [l for l in self.allocator.free_labels if self.depth[l] < deadline]
        if eligible:
            label = min(eligible, key=lambda l: self.depth[l])
            self.allocator.reuse(label)
        else:
            label = self.allocator.allocate_fresh()
        return label

    def free(self, qubit):
        self.allocator.free(qubit)

    def next_step(self):
        self.allocator.next_step()

    def postselect(self, qubit, state):
        self.add_gate("PostSelect", qubit, state = state)
        self.free(qubit)

    def init_qubit(self, qubit, state):
        self.add_gate("InitAncilla", qubit, state = state)

    def split_small(self, source_qubit, split_on_Z=True, deadline=None):
        d_src = self.depth[source_qubit]
        if deadline is not None:
            d_src = max(d_src, deadline)
    
        eligible = [l for l in self.allocator.free_labels if self.depth[l] < d_src]
        if eligible:
            ancilla1 = min(eligible, key=lambda l: self.depth[l])
            self.allocator.reuse(ancilla1)
        else:
            ancilla1 = self.allocator.allocate_fresh()
            # fresh label has depth 0 -> its init would be scheduled at column 0 and
            # dangle until d_src. Pin it one column before its first use instead.
            self.depth[ancilla1] = max(0, d_src - 1)
    
        if split_on_Z:
            self.init_qubit(ancilla1, "0")
            self.add_gate("CNOT", source_qubit, ancilla1)
        else:
            self.init_qubit(ancilla1, "+")
            self.add_gate("CNOT", ancilla1, source_qubit)
        self.row = ancilla1; self.pivot = source_qubit
        self.history.append(("split", source_qubit, ancilla1))
        return source_qubit, ancilla1
    
    
    def add_measurement(self, qubit1, qubit2, spider_type, keep_order=False):
        # the connection is symmetric: take the leg off the end that is ready first
        if not keep_order and self.depth[qubit2] < self.depth[qubit1]:
            qubit1, qubit2 = qubit2, qubit1

        # split must land no earlier than one column before the merge target is free
        deadline = max(self.depth[qubit1], self.depth[qubit2] - 1)
    
        if spider_type == "X":
            _, out = self.split_small(qubit1, split_on_Z=False, deadline=deadline)
            self.merge_small(out, qubit2, merge_on_X=True)
        else:
            _, out = self.split_small(qubit1, split_on_Z=True, deadline=deadline)
            self.merge_small(out, qubit2, merge_on_X=False)

    def merge_small(self, qubit1, qubit2, merge_on_X = True):
        if merge_on_X:
            self.add_gate("CNOT", qubit1, qubit2)
            self.postselect(qubit1, "+")
        else:
            self.add_gate("CNOT", qubit2, qubit1)
            self.postselect(qubit1, "0")

        self.history.append(("merge", qubit1, "into", qubit2))
        return qubit2

    # def split_small(self, source_qubit, split_on_Z = True):
    #     d_src = self.depth[source_qubit]
    #     # reuse a freed wire only if it is idle strictly before the source is ready
    #     # (depth[l] < d_src ⇒ its re-init CNOT lands at d_src, no extra column)
    #     eligible = [l for l in self.allocator.free_labels if self.depth[l] < d_src]
    #     # print(eligible)
    #     if eligible:
    #         ancilla1 = min(eligible, key = lambda l: self.depth[l])
    #         self.allocator.reuse(ancilla1)
    #     else:
    #         ancilla1 = self.allocator.allocate_fresh()
    #     if split_on_Z:
    #         self.init_qubit(ancilla1, "0")
    #         self.add_gate("CNOT", source_qubit, ancilla1)
    #     else:
    #         self.init_qubit(ancilla1, "+")
    #         self.add_gate("CNOT", ancilla1, source_qubit)
    #     self.row = ancilla1; self.pivot = source_qubit
    #     self.history.append(("split", source_qubit, ancilla1))

    #     return source_qubit, ancilla1

    # def add_measurement(self, qubit1, qubit2, spider_type):
    #     # deadline = max(self.depth[qubit1], self.depth[qubit2])
    #     if spider_type == "X":
    #         _, out = self.split_small(qubit1, split_on_Z = True)
    #         self.merge_small(out, qubit2, merge_on_X = False)
    #     else:
    #         _, out = self.split_small(qubit1, split_on_Z = True)
    #         self.merge_small(out, qubit2, merge_on_X = False)
    #     # self.next_step()

    @staticmethod
    def _apply_stretch_columns(g, at_col, delta):
        for v in g.vertices():
            if g.row(v) > at_col:
                g.set_row(v, g.row(v) + delta)

    @staticmethod
    def _apply_stretch_wire(g, q, at_col, delta):
        for v in g.vertices():
            if g.qubit(v) == q and g.row(v) > at_col:
                g.set_row(v, g.row(v) + delta)

    @staticmethod
    def _normalize_stretch(s):
        if len(s) == 3:
            q, at_col, delta = s
            return ("wire", q, at_col, delta)
        return tuple(s)

    def _apply_stretch(self, g, s):
        kind = s[0]
        if kind == "wire":
            _, q, at_col, delta = s
            self._apply_stretch_wire(g, q, at_col, delta)
        elif kind == "cols":
            _, _, at_col, delta = s
            self._apply_stretch_columns(g, at_col, delta)

    # ---- stretch bookkeeping ---------------------------------------------------------------------------------

    def stretch_wire(self, q, at_col, delta):
        self.stretches.append(("wire", q, at_col, delta))
        return self

    def stretch_columns(self, at_col, delta):
        self.stretches.append(("cols", None, at_col, delta))
        return self

    def clear_stretches(self):
        self.stretches = []
        return self

    @staticmethod
    def _normalize_stretch(s):
        if len(s) == 3:         
            q, at_col, delta = s
            return ("wire", q, at_col, delta)
        return tuple(s)

    @staticmethod
    def _apply_stretch_columns(g, at_col, delta):
        for v in g.vertices():
            if g.row(v) > at_col:
                g.set_row(v, g.row(v) + delta)

    # ---- layout ----------------------------------------------------------------------------------------------

    @staticmethod
    def _gate_groups(g):
        parent = {v: v for v in g.vertices()}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for e in g.edges():
            u, v = g.edge_st(e)
            if g.qubit(u) != g.qubit(v) and g.row(u) == g.row(v):
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv

        members = defaultdict(list)
        for v in g.vertices():
            members[find(v)].append(v)
        return members

    def _relayout(self, g, wire_pads, tight=False):
        members = self._gate_groups(g)
        group_of = {v: root for root, vs in members.items() for v in vs}

        by_wire = defaultdict(list)
        for v in g.vertices():
            by_wire[g.qubit(v)].append(v)
        for q in by_wire:
            by_wire[q].sort(key=lambda v: g.row(v))

        # a stretch becomes "extra gap in front of the first gate on this wire after at_col"
        pad = defaultdict(int)
        for (q, at_col), delta in wire_pads.items():
            for v in by_wire.get(q, ()):
                if g.row(v) > at_col:
                    pad[(q, group_of[v])] += delta
                    break

        # the packed layout is a valid schedule, so its column order is a topological order
        order = sorted(members, key=lambda root: min(g.row(v) for v in members[root]))

        # a group moves only if a pad, or a predecessor pushed by a pad, forces it.
        # boundaries get no special treatment, so inputs/outputs keep the columns
        # repack_gate gave them unless something on their own wire pushes into them.
        new_row, last = {}, {}
        for root in order:
            vs = members[root]
            col = 0 if tight else min(g.row(v) for v in vs)
            for v in vs:
                q = g.qubit(v)
                prev = last.get(q)
                col = max(col, (0 if prev is None else prev + 1) + pad[(q, root)])
            new_row[root] = col
            for v in vs:
                last[g.qubit(v)] = col

        for root, col in new_row.items():
            for v in members[root]:
                g.set_row(v, col)

    def build_graph(self, stretches=None, tight=False, align_outputs=True):
        g = self.circ.to_graph()
        repack_gate(g)

        wire_pads = defaultdict(int)
        col_shifts = []
        for s in (self._normalize_stretch(x) for x in self.stretches + list(stretches or [])):
            if s[0] == "wire":
                _, q, at_col, delta = s
                wire_pads[(q, at_col)] += delta
            elif s[0] == "cols":
                col_shifts.append(s)

        # if wire_pads:
        #     self._relayout(g, dict(wire_pads), tight=tight)
        # for _, _, at_col, delta in col_shifts:            # rigid, already alignment-safe
        #     self._apply_stretch_columns(g, at_col, delta)

        if wire_pads or tight:
            self._relayout(g, dict(wire_pads), tight=tight)
        if tight and align_outputs:
            self._align_outputs(g)
        for _, _, at_col, delta in col_shifts:
            self._apply_stretch_columns(g, at_col, delta)
        return g

        return g

    def draw(self, stretches = None, tight = False):
        zx.draw(self.build_graph(stretches, tight=tight))

    def get_circuit(self):
        return self.circ

    def allocator_state(self):
        return self.allocator.state()

    def merge_many_to_one(self, qubits, spider_type, merge_up = False, verbose = False):
        if verbose:
            print("(MERGE) BEFORE", self.allocator.state())

        if not qubits:
            raise ValueError("cannot merge zero qubits")

        merge_on_X = spider_type == "X"
        pivot = qubits[0]

        current = qubits[0]
        iterator = 0
        for q in qubits[1:]:
            if iterator >= 1:
                # print("HERE")
                # print(self.allocator.state())
                self.next_step()   
            # current = self.merge_small(current, q, merge_on_X)
            # pivot = self.merge_small(pivot, q, merge_on_X)
            if merge_up:
                pivot = self.merge_small(q, pivot, merge_on_X = merge_on_X)
            else:
                pivot = self.merge_small(pivot, q, merge_on_X = merge_on_X)
            iterator += 1

        if verbose:
            print("(MERGE) AFTER", self.allocator.state())

        return pivot

    @staticmethod
    def _align_outputs(g):
        outs = g.outputs() if callable(getattr(g, "outputs", None)) else g.outputs
        outs = set(outs)
        if not outs:
            return
        last = max((g.row(v) for v in g.vertices() if v not in outs), default=0)
        for v in outs:
            g.set_row(v, last + 1)
        
    @staticmethod
    def _validate_split_request(n_outputs, spider_type):
        if n_outputs < 1:
            raise ValueError("n_outputs must be >= 1")
        if spider_type not in ("Z", "X"):
            raise ValueError("spider_type must be 'Z' or 'X'")

    def _split_from_pivot(self, pivot, n_outputs, spider_type):
        split_on_Z = spider_type == "Z"
        outputs = [pivot]

        while len(outputs) < n_outputs:
            source = outputs.pop()
            left, right = self.split_small(source, split_on_Z = split_on_Z)
            outputs.extend([left, right])
            self.next_step()

        return outputs[:n_outputs]

    def split_one_to_many(self, pivot, n_outputs, spider_type, verbose = False):
        if verbose:
            print("(SPLIT) BEFORE", self.allocator.state())

        self._validate_split_request(n_outputs, spider_type)
        if not self.allocator.is_active(pivot):
            raise ValueError(f"pivot qubit {pivot} is not active")

        # Move allocator quarantine forward once before this spider step.
        self.next_step()
        outputs = self._split_from_pivot(pivot, n_outputs, spider_type)

        if verbose:
            print("(SPLIT) AFTER", self.allocator.state())

        return outputs

    # def split_zero_to_many(self, n_outputs, spider_type, phase = 0, verbose = False):
    #     if verbose:
    #         print("(ZERO-SPLIT) BEFORE", self.allocator.state())

    #     self._validate_split_request(n_outputs, spider_type)

    #     self.next_step()
    #     pivot = self.allocate()
    #     self.init_qubit(pivot, "+" if spider_type == "Z" else "0")
    #     self.apply_spider(pivot, spider_type, phase)

    #     outputs = self._split_from_pivot(pivot, n_outputs, spider_type)

    #     if verbose:
    #         print("(ZERO-SPLIT) AFTER", self.allocator.state())

    #     return outputs
    def split_zero_to_many(self, n_outputs, spider_type, phase = 0, start_col = 0, verbose=False):
        if verbose:
            print("(ZERO-SPLIT) BEFORE", self.allocator.state())
    
        self._validate_split_request(n_outputs, spider_type)
        self.next_step()
    
        if start_col is None:
            # reuse whichever free wire goes idle earliest, wherever that lands
            pivot = self._allocate_before(float("inf"))
        else:
            # the new spider has to fit at start_col, so only reuse a wire that is
            # already idle by then; otherwise take a new row
            pivot = self._allocate_before(start_col + 1)
            self.depth[pivot] = max(self.depth[pivot], start_col)
    
        self.init_qubit(pivot, "+" if spider_type == "Z" else "0")
        self.apply_spider(pivot, spider_type, phase)
        outputs = self._split_from_pivot(pivot, n_outputs, spider_type)
    
        if verbose:
            print("(ZERO-SPLIT) AFTER", self.allocator.state())
    
        return outputs

    def apply_spider(self, qubit, spider_type, phase):
        if phase == 0:
            return qubit

        if spider_type == "Z":
            self.add_gate("ZPhase", qubit, phase = phase)
        elif spider_type == "X":
            self.add_gate("XPhase", qubit, phase = phase)
        elif spider_type == "BOUNDARY":
            pass

        return qubit