from __future__ import annotations
import math
import random
import sys
from fractions import Fraction
from pathlib import Path
import pyzx as zx
import pyzx.simplify as simplify
import stim
from pyzx.utils import EdgeType, VertexType

from spidercat_shor import CatLibrary

# PASTED FROM THE IPYNB

class StimToPyZXExtractor:
    SINGLE = {"I": None,
              "X": (VertexType.X, Fraction(1)),
              "Z": (VertexType.Z, Fraction(1)),
              "S": (VertexType.Z, Fraction(1, 2)),
              "S_DAG": (VertexType.Z, Fraction(-1, 2))}
    SINGLE_MULTI = {"H": [(VertexType.H_BOX, None)],
                    "Y": [(VertexType.Z, Fraction(1)), (VertexType.X, Fraction(1))]}
    CNOT_LIKE = {"CX", "CNOT", "ZCX"}
    CZ_LIKE = {"CZ", "ZCZ"}
    CY_LIKE = {"CY", "ZCY"}
    MEAS = {"M", "MZ", "MX", "MY"}
    RESET = {"R", "RZ", "RX", "RY"}
    MEAS_RESET = {"MR", "MRZ", "MRX", "MRY"}
    NOOP = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
    INIT_OPTS = {"open", "zero", "plus"}

    # how a prep spider is colored
    # "state" -> color reflects the prepared state (red = |0>, green = |+>)
    # "match" -> recolor to the nearest spider's color; this FLIPS the represented state (|0><->|+>) on misatched lines
    # "match_keep_state" -> recolour to the nearest spiders color but flip the connecting edge (simple<->Hadamard) so the prepared
    #                       state is unchanged
    PREP_COLORS = {"state", "match", "match_keep_state"}

    def __init__(self, open_inputs = True, max_reps = 1, postselect = True, init = None, prep_color = "state",
                 serialize_cnot = False, init_all_left = False, avoid_postselection_crossings = False, z_only = False,
                 cat_library: CatLibrary | None = None, t: int | None = None, spidercat_hub: bool = False,
                 min_weight: int = 4, strict_hub: bool = False):

        self.cat_library = cat_library
        self.cat_t = t
        self.spidercat_hub = spidercat_hub
        self.min_weight = min_weight
        self.strict_hub = strict_hub

        if init is None:
            init = "open" if open_inputs else "zero"
        if init not in self.INIT_OPTS:
            raise ValueError(f"init must be one of {sorted(self.INIT_OPTS)}")
        if prep_color not in self.PREP_COLORS:
            raise ValueError(f"prep_color must be one of {sorted(self.PREP_COLORS)}")
        self.init = init
        self.prep_color = prep_color
        self.open_inputs = init == "open"
        self.max_reps = max_reps
        self.postselect = postselect
        self.serialize_cnot = serialize_cnot
        self.init_all_left = init_all_left
        self.avoid_postselection_crossings = avoid_postselection_crossings
        self.z_only = z_only
        self.init_state(None)


    def init_state(self, circuit):
        self.circuit = circuit
        self.n = circuit.num_qubits if circuit is not None else 0
        self.g = zx.graph.Graph()
        self.frontier = {q: None for q in range(self.n)}
        self.pending = {}
        self.inputs = []
        self.outputs = []
        self.meas_qubits = []
        self.mpp_products = []
        self.skipped = {"detector": 0, "observable": 0}
        self.preps = [] 

        self._clock = 1
        self._hub_counter = 0

        self.postselection_sites = set()
        self.cnot_target_spiders = set()

        self.cat_hubs = []
        self._cat_lane = 0

    # this is whay i wanted to do in the actual extractor
    # might eaither copy or switch extraction to STIM -> pyZX
    def alloc(self):
        c = self._clock
        self._clock += 1
        return c

    @staticmethod
    def sublayers(items, qubits_of):
        layers = []
        last_layer = {}

        for item in items:
            qubits = tuple(qubits_of(item))
            layer_index = 1 + max((last_layer.get(q, -1) for q in qubits), default = -1)
            while len(layers) <= layer_index:
                layers.append([])
            layers[layer_index].append(item)
            for q in qubits:
                last_layer[q] = layer_index
        return layers

    # return True if any CNOT edge at row would cross or overlap a post-selection measurement spider
    def postsel_hit(self, pairs, row):
        if not self.avoid_postselection_crossings:
            return False
        for measured_q, measured_row in self.postselection_sites:
            if measured_row != row:
                continue
            for control, target in pairs:
                low = min(control, target)
                high = max(control, target)
                # inclusive so that an overlapping endpoint is also rejected? (edge case maybve)
                if low <= measured_q <= high:
                    return True

        return False

    def alloc_cnot_row(self, pairs):
        row = self.alloc()
        while self.postsel_hit(pairs, row):
            row = self.alloc()
        return row

    def spider(self, q, vtype, phase, col):
        v = self.g.add_vertex(vtype, qubit = q, row = col, phase = phase)
        if self.frontier[q] is not None:
            self.g.add_edge((self.frontier[q], v), EdgeType.SIMPLE)
        self.frontier[q] = v
        return v

    def absorb_cnot_target_postselection(self, q):
        if not self.z_only:
            return False

        target_spider = self.frontier[q]

        if target_spider not in self.cnot_target_spiders:
            return False

        if self.g.type(target_spider) != VertexType.X:
            return False

        self.g.set_type(target_spider, VertexType.Z)
        self.cnot_target_spiders.discard(target_spider)

        return True

    def prepare_zero_at(self, q, col, hadamard = False):
        v = self.g.add_vertex(VertexType.Z if hadamard else VertexType.X, qubit = q, row = col, phase = Fraction(0))
        self.frontier[q] = v
        self.preps.append(v)

    def ensure_layer(self, involved):
        need = [q for q in involved if self.frontier[q] is None]
        if need:
            c = self.alloc()
            for q in need:
                self.prepare_zero_at(q, c, hadamard = self.pending.pop(q, False))

    def recolor_preps(self):
        if self.prep_color == "state":
            return
        keep = (self.prep_color == "match_keep_state")
        Z, X = VertexType.Z, VertexType.X
        SIMPLE, HAD = EdgeType.SIMPLE, EdgeType.HADAMARD
        for v in self.preps:
            nbrs = list(self.g.neighbors(v))
            if len(nbrs) != 1:
                continue
            w = nbrs[0]
            wt = self.g.type(w)
            if wt not in (Z, X) or self.g.type(v) == wt:
                # neightbour is a boundary (H) or aready matches
                continue              
            self.g.set_type(v, wt)
            if keep:
                e = self.g.edge(v, w)
                et = self.g.edge_type(e)
                self.g.set_edge_type(e, HAD if et == SIMPLE else SIMPLE)

    @staticmethod
    def parse_pauli_products(targets):
        products, current, prev_combiner = [], [], False
        for t in targets:
            if t.is_combiner:
                prev_combiner = True
                continue
            if current and not prev_combiner:
                products.append(current)
                current = []
            if t.is_x_target:
                p = "X"
            elif t.is_y_target:
                p = "Y"
            elif t.is_z_target:
                p = "Z"
            else:
                raise NotImplementedError("MPP target is not a Pauli")
            current.append((t.qubit_value, p, t.is_inverted_result_target))
            prev_combiner = False
        if current:
            products.append(current)
        return products

    @staticmethod
    def pairs(qs):
        return [(qs[i], qs[i + 1]) for i in range(0, len(qs), 2)]

    def prep_if_needed(self, qubits, col):
        for q in qubits:
            if self.frontier[q] is None:
                self.prepare_zero_at(q, col, hadamard = self.pending.pop(q, False))

    def do_cz(self, qs):
        pairs = self.pairs(qs)
        groups = [[p] for p in pairs] if self.serialize_cnot else self.sublayers(pairs, lambda p: p)
        for grp in groups:
            c = self.alloc_cnot_row(grp)
            for a, b in grp:
                self.prep_if_needed((a, b), c - 1)
                va = self.spider(a, VertexType.Z, Fraction(0), c)
                vb = self.spider(b, VertexType.Z, Fraction(0), c)
                self.g.add_edge((va, vb), EdgeType.HADAMARD)

    def do_cy(self, qs):
        pairs = self.pairs(qs)
        groups = [[p] for p in pairs] if self.serialize_cnot else self.sublayers(pairs, lambda p: p)
        for grp in groups:
            c0 = self.alloc()
            c1 = self.alloc_cnot_row(grp)
            c2 = self.alloc()
            for control, target in grp:
                self.prep_if_needed((control, target), c0 - 1)
                self.spider(target, VertexType.Z, Fraction(-1, 2), c0)
                zc = self.spider(control, VertexType.Z, Fraction(0), c1)
                xt = self.spider(target, VertexType.X, Fraction(0), c1)
                self.g.add_edge((zc, xt), EdgeType.SIMPLE)
                self.cnot_target_spiders.add(xt)
                self.spider(target, VertexType.Z, Fraction(1, 2), c2)

    
    def add_marked_graph(self, G, M, center, neg):
        lane = self.n + 1 + self._cat_lane
        self._cat_lane += max(4, G.number_of_nodes() // 2)

        vmap = {}
        for i, v in enumerate(sorted(G.nodes())):
            vmap[v] = self.g.add_vertex(VertexType.Z, qubit = lane + 1 + (i % 6),
                                        row = center - 1 + (i // 6), phase = Fraction(0))
        if neg:
            # every Z-spider in the graph fuses, so one pi anywhere is the whole phase
            self.g.set_phase(vmap[sorted(G.nodes())[0]], Fraction(1))

        legs = []
        for k, (u, v) in enumerate(sorted(tuple(sorted(e)) for e in G.edges())):
            marks = M.get((u, v), 0)
            if marks == 0:
                self.g.add_edge((vmap[u], vmap[v]), EdgeType.SIMPLE)
                continue
            prev = vmap[u]
            for j in range(marks):
                b = self.g.add_vertex(VertexType.Z, qubit = lane + 1 + (k % 6) + 0.5,
                                      row = center - 1 + (k // 6) + 0.4 + 0.2 * j,
                                      phase = Fraction(0))
                self.g.add_edge((prev, b), EdgeType.SIMPLE)
                prev = b
                legs.append(b)
            self.g.add_edge((prev, vmap[v]), EdgeType.SIMPLE)
        return legs, list(vmap.values())

    def resolve_cat_graph(self, n):
        if not self.spidercat_hub or n < self.min_weight:
            return None
        t = self.cat_t if self.cat_t is not None else 1
        graph = self.cat_library.marked_graph(n, t) if self.cat_library is not None else None
        if graph is None:
            msg = (f"no SpiderCat marked graph for n = {n}, t = {t}; using the idealised "
                   f"single-spider hub. Pre-generate it with "
                   f"`python generate.py -n {n} -t {t}`, or pass strict_hub = True to raise.")
            if self.strict_hub:
                raise LookupError(msg)
            print("warning:", msg, file = sys.stderr)
        return graph

    def measure_pauli_product(self, product):
        n = len(product)
        graph = self.resolve_cat_graph(n)

        qubits = [q for q, _, _ in product]
        self.ensure_layer(qubits)
        neg = sum(1 for _, _, inv in product if inv) % 2
        depth_of = {"Z": 0, "X": 1, "Y": 2}
        depth = max(depth_of[p] for _, p, _ in product)
        pre_cols = [self.alloc() for _ in range(depth)]
        center = self.alloc()
        post_cols = [self.alloc() for _ in range(depth)]
        self.meas_qubits.extend(qubits)
        self.mpp_products.append({"terms": list(product), "inverted": bool(neg)})

        if graph is None:
            # idealised specification: one X-spider of arity n
            hub_row = self.n
            hub = self.g.add_vertex(VertexType.X, qubit = hub_row, row = center,
                                    phase = Fraction(1) if neg else Fraction(0))
            legs = [hub] * n
            leg_edge = EdgeType.SIMPLE
        else:
            # fault-tolerant replacement: a 3-regular marked graph, one leg per mark
            G, M, t_achieved, source = graph
            legs, vertex_ids = self.add_marked_graph(G, M, center, neg)
            leg_edge = EdgeType.HADAMARD
            if len(legs) != n:
                raise AssertionError(f"marked graph has {len(legs)} marks "
                                     f"but the product has weight {n}")

        for (q, p, _), leg in zip(product, legs):
            if p == "Z":
                det = self.spider(q, VertexType.Z, Fraction(0), center)
            elif p == "X":
                self.spider(q, VertexType.H_BOX, None, pre_cols[-1])
                det = self.spider(q, VertexType.Z, Fraction(0), center)
                self.spider(q, VertexType.H_BOX, None, post_cols[0])
            else:
                self.spider(q, VertexType.Z, Fraction(-1, 2), pre_cols[-2])
                self.spider(q, VertexType.H_BOX, None, pre_cols[-1])
                det = self.spider(q, VertexType.Z, Fraction(0), center)
                self.spider(q, VertexType.H_BOX, None, post_cols[0])
                self.spider(q, VertexType.Z, Fraction(1, 2), post_cols[1])
            if self.postselect:
                self.postselection_sites.add((q, center))
            self.g.add_edge((det, leg), leg_edge)

        if graph is None:
            if not self.postselect:
                col = post_cols[-1] if post_cols else center
                b = self.g.add_vertex(VertexType.BOUNDARY, qubit = hub_row, row = col + 1)
                self.g.add_edge((hub, b), EdgeType.SIMPLE)
                self.outputs.append(b)
        else:
            self.cat_hubs.append({"terms": list(product), "weight": n, "t": t_achieved,
                                  "source": source, "num_vertices": G.number_of_nodes(),
                                  "num_edges": G.number_of_edges(), "leg_vertices": list(legs),
                                  "graph_vertices": list(vertex_ids),
                                  "vertex_ratio": G.number_of_nodes() / n})

    def process(self, instrs):
        for inst in instrs:
            if isinstance(inst, stim.CircuitRepeatBlock):
                reps = (inst.repeat_count if self.max_reps is None else min(inst.repeat_count, self.max_reps))
                for _ in range(reps):
                    self.process(inst.body_copy())
                continue
            name = inst.name
            qs = [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]
            if name in self.NOOP:
                if name == "DETECTOR": self.skipped["detector"] += 1
                if name == "OBSERVABLE_INCLUDE": self.skipped["observable"] += 1
                continue
            if name in self.SINGLE:
                spec = self.SINGLE[name]
                if spec is not None:
                    for grp in self.sublayers(qs, lambda q: (q,)):
                        self.ensure_layer(grp)
                        c = self.alloc()
                        for q in grp:
                            self.spider(q, spec[0], spec[1], c)
            elif name in self.SINGLE_MULTI:
                seq = self.SINGLE_MULTI[name]
                for grp in self.sublayers(qs, lambda q: (q,)):
                    self.ensure_layer(grp)
                    cols = [self.alloc() for _ in seq]
                    for q in grp:
                        for (vt, ph), c in zip(seq, cols):
                            self.spider(q, vt, ph, c)
            elif name in self.CNOT_LIKE:
                pairs = [(qs[i], qs[i + 1]) for i in range(0, len(qs), 2)]

                if self.serialize_cnot:
                    groups = [[pair] for pair in pairs]
                else:
                    groups = self.sublayers(pairs, lambda pair: pair)

                for grp in groups:
                    c = self.alloc_cnot_row(grp)

                    for control, target in grp:
                        target_is_fresh_zero = (self.frontier[target] is None and not self.pending.get(target, False))
                        use_z_unfusion = (self.z_only and target_is_fresh_zero)

                        if use_z_unfusion:
                            if self.frontier[control] is None:
                                self.prepare_zero_at(control, c - 1, hadamard = self.pending.pop(control, False))

                            self.pending.pop(target, None)
                            z_control = self.spider(control, VertexType.Z, Fraction(0), c)
                            z_target = self.spider(target, VertexType.Z, Fraction(0), c)

                            self.g.add_edge((z_control, z_target), EdgeType.SIMPLE)

                        else:
                            for q in (control, target):
                                if self.frontier[q] is None:
                                    self.prepare_zero_at(q, c - 1, hadamard=self.pending.pop(q, False))

                            z_control = self.spider(control, VertexType.Z, Fraction(0), c)
                            x_target = self.spider(target, VertexType.X, Fraction(0), c)

                            self.g.add_edge((z_control, x_target), EdgeType.SIMPLE)

                            self.cnot_target_spiders.add(x_target)
            elif name in self.CZ_LIKE:
                self.do_cz(qs)
            elif name in self.CY_LIKE:
                self.do_cy(qs)
            elif name == "SWAP":
                for i in range(0, len(qs), 2):
                    a, b = qs[i], qs[i + 1]
                    self.ensure_layer([a, b])
                    self.frontier[a], self.frontier[b] = self.frontier[b], self.frontier[a]
            elif name == "MPP":
                for prod in self.parse_pauli_products(inst.targets_copy()):
                    self.measure_pauli_product(prod)
            elif name in self.MEAS or name in self.MEAS_RESET:
                if name.endswith("Y"): raise NotImplementedError("Y-basis measure/reset not supported")
                had = name.endswith("X")
                for grp in self.sublayers(qs, lambda q: (q,)):
                    self.ensure_layer(grp)
                    c = None if self.postselect else self.alloc()
                    for q in grp:
                        self.meas_qubits.append(q)
                        if self.postselect:
                            previous = self.frontier[q]
                            if (not had and self.absorb_cnot_target_postselection(q)):
                                absorbed_row = self.g.row(previous)
                                self.postselection_sites.add((q, absorbed_row))
                                self.frontier[q] = None
                                self.pending[q] = False
                                continue
                            col = self.g.row(self.frontier[q]) + 1
                            measurement = self.spider(q, (VertexType.Z if name.endswith("X") else VertexType.X), 
                                                      Fraction(0), col)
                            self.postselection_sites.add((q, col))
                            self.frontier[q] = None
                            self.pending[q] = had  
                        else:
                            b = self.g.add_vertex(VertexType.BOUNDARY, qubit = q, row = c)
                            self.g.add_edge((self.frontier[q], b), EdgeType.HADAMARD if had else EdgeType.SIMPLE)
                            self.outputs.append(b)
                            self.frontier[q] = None
                            if name in self.MEAS_RESET:
                                self.pending[q] = had
            elif name in self.RESET:
                had = name.endswith("X")
                c = None
                for q in qs:
                    if self.frontier[q] is not None:
                        if self.postselect:
                            col = self.g.row(self.frontier[q]) + 1
                            self.spider(q, VertexType.X, Fraction(0), col)
                        else:
                            if c is None: c = self.alloc()
                            b = self.g.add_vertex(VertexType.BOUNDARY, qubit = q, row = c)
                            self.g.add_edge((self.frontier[q], b), EdgeType.SIMPLE)
                            self.outputs.append(b)
                        self.frontier[q] = None
                    self.pending[q] = had
            else:
                raise NotImplementedError(f"gate {name!r} not supported yet")

    def extract(self, circuit):
        if isinstance(circuit, str):
            circuit = stim.Circuit(circuit)
        self.init_state(circuit)
        if self.init == "open":
            c = self.alloc()
            for q in range(self.n):
                b = self.g.add_vertex(VertexType.BOUNDARY, qubit = q, row = c)
                self.inputs.append(b)
                self.frontier[q] = b
        else:
            hadamard = self.init == "plus"
            if self.init_all_left:
                for q in range(self.n):
                    self.prepare_zero_at(q, col = 0, hadamard = hadamard)
            else:
                for q in range(self.n):
                    self.pending[q] = hadamard

        self.process(self.circuit)
        live = [q for q in range(self.n) if self.frontier[q] is not None]
        if live:
            c = self.alloc()
            for q in live:
                b = self.g.add_vertex(VertexType.BOUNDARY, qubit = q, row = c)
                self.g.add_edge((self.frontier[q], b), EdgeType.SIMPLE)
                self.outputs.append(b)
        self.recolor_preps()
        self.g.set_inputs(tuple(self.inputs))
        self.g.set_outputs(tuple(self.outputs))
        return self.g, {"measured_qubits": self.meas_qubits, "mpp": self.mpp_products,
                        "skipped": self.skipped, "cat_hubs": self.cat_hubs}

def stim_to_pyzx(circuit, open_inputs = True, max_reps = 1, postselect = True, init = None,
                 prep_color = "match", serialize_cnot = True, init_all_left = False,
                 avoid_postselection_crossings = True, z_only = False,
                 cat_library: CatLibrary | None = None, t: int | None = None,
                 spidercat_hub: bool = False, min_weight: int = 4, strict_hub: bool = False):
    return StimToPyZXExtractor(open_inputs = open_inputs, max_reps = max_reps, postselect = postselect,
                               init = init, prep_color = prep_color, serialize_cnot = serialize_cnot,
                               init_all_left = init_all_left,
                               avoid_postselection_crossings = avoid_postselection_crossings,
                               z_only = z_only, cat_library = cat_library, t = t,
                               spidercat_hub = spidercat_hub, min_weight = min_weight,
                               strict_hub = strict_hub).extract(circuit)


def mpp_to_cat_zx(mpp_circuit, library: CatLibrary, t: int | None = None, distance: int | None = None, min_weight: int = 4, **kwargs):
    if t is None:
        if distance is None:
            raise ValueError("give t = ... or distance = ...")
        t = distance // 2
    if isinstance(mpp_circuit, Path) or (isinstance(mpp_circuit, str) 
                                         and "\n" not in mpp_circuit and len(mpp_circuit) < 4096 and Path(mpp_circuit).is_file()):
        mpp_circuit = stim.Circuit(Path(mpp_circuit).read_text())
    return stim_to_pyzx(mpp_circuit, cat_library = library, t = t, spidercat_hub = True,
                        min_weight = min_weight, **kwargs)


def save_zx(g, stem, png = False, figsize = (14, 6), dpi = 150):
    from pathlib import Path
 
    stem = Path(stem)
    stem.parent.mkdir(parents = True, exist_ok = True)
    written = []
    for suffix, text in ((".zxg", g.to_json()), (".tikz", g.to_tikz())):
        p = stem.with_suffix(suffix)
        p.write_text(text)
        written.append(p)
    if png:
        fig = zx.draw_matplotlib(g, figsize = figsize)
        p = stem.with_suffix(".png")
        fig.savefig(p, dpi = dpi, bbox_inches = "tight")
        written.append(p)
    return written