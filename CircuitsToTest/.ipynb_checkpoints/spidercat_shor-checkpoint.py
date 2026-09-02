"""
replace high-weight MPP instructions in stim syndrome-extraction circuits by shor-style measurements that use fault-tolerant CAT states from SpiderCat (arXiv:2603.05391, https://github.com/boldar99/spidercat)

the gadget for MPP P0*P1*...*P{n-1} is:
    1. prepare |CAT_n> = |0..0> + |1..1>  on n fresh ancillas  (SpiderCat circuit, flag qubits measured + post-selected via DETECTOR(s),
    2. for every term, a controlled-Pauli from the cat leg to the data qubit:
           X_q  ->  CX  leg q 
           Y_q  ->  CY  leg q
           Z_q  ->  CZ  leg q
    3. MX on all n legs -> the XOR of those n outcomes is the syndrome bit.

sanity check for the wiring: 
|CAT_1> = |+>, 
and CX a q 
MX a measures X_q
"""

from __future__ import annotations
import json
import re
import itertools
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import stim

# whole bunch of stuff
__all__ = ["PauliTerm", "CatState", "CatLibrary", "MppCatRewriter", "RewriteResult", 
           "parse_pauli_products", "find_spidercat_root",  "code_params_from_name"]

MEASUREMENT_GATES = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY", "MPP"}
CONTROLLED = {"X": "CX", "Y": "CY", "Z": "CZ"}
_TWO_QUBIT_GATES = {"CX", "CNOT", "ZCX", "XCZ", "CY", "ZCY", "YCZ", "CZ", "ZCZ",
                    "SWAP", "ISWAP", "ISWAP_DAG", "XCX", "XCY", "YCX", "YCY", "SQRT_XX", "SQRT_ZZ"}
_ENTANGLING = _TWO_QUBIT_GATES
_RESET_GATES = {"R", "RX", "RY", "RZ"}
_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}


# Paulis
@dataclass(frozen = True)
class PauliTerm:
    qubit: int
    pauli: str 
    inverted: bool = False
    def __str__(self) -> str:
        return f"{'!' if self.inverted else ''}{self.pauli}{self.qubit}"

def parse_pauli_products(targets: Sequence[stim.GateTarget]) -> list[list[PauliTerm]]:
    products: list[list[PauliTerm]] = []
    current: list[PauliTerm] = []
    prev_combiner = False
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
            raise NotImplementedError("MPP target is not a Pauli???")
        current.append(PauliTerm(t.qubit_value, p, bool(t.is_inverted_result_target)))
        prev_combiner = False
    if current:
        products.append(current)
    return products

def pauli_product_str(terms: Sequence[PauliTerm]) -> str:
    return "*".join(str(t) for t in terms)


def t_cap(n: int) -> int:
    # largest meaningful fault-tolerance for an n-qubit CAT state
    # SpiderCat's generate.cat_state_FT uses T = min(t, floor(n/2) - 1): 
    # a cut can always split the marks roughly in half, so asking for more is not achievable the shipped circuits reflect this 
    return max(0, n // 2 - 1)

@dataclass
class CatState:
    # a fault-tolerant CAT-state preparation circuit plus its leg/flag structure
    circuit: stim.Circuit
    n: int                      
    t: int                     
    t_requested: int
    legs: tuple[int, ...]      
    num_flag_measurements: int
    source: str                

    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits

    @property
    def num_flags(self) -> int:
        return self.num_qubits - self.n

    @property
    def capped(self) -> bool:
        return self.t < self.t_requested


def measured_qubits(circuit: stim.Circuit) -> set[int]:
    out: set[int] = set()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out |= measured_qubits(op.body_copy())
            continue
        if op.name in MEASUREMENT_GATES:
            out |= {t.qubit_value for t in op.targets_copy() if t.is_qubit_target}
    return out


def postselect_uncovered_measurements(circuit: stim.Circuit) -> stim.Circuit:
    # append DETECTOR rec[-k] for flag measurements that no DETECTOR covers cat_state_6() for instance ends in a bare M 0 1
    # without detectors those flags are not post-selected and the gadget silently loses its fault tolerance (I THINK)
    n_meas = circuit.num_measurements
    if n_meas == 0:
        return circuit
    covered: set[int] = set()
    seen = 0
    for op in circuit.flattened():
        if op.name in MEASUREMENT_GATES:
            seen += sum(1 for t in op.targets_copy() if t.is_qubit_target)
        elif op.name == "DETECTOR":
            for t in op.targets_copy():
                if t.is_measurement_record_target:
                    covered.add(seen + t.value)
    missing = [i for i in range(n_meas) if i not in covered]
    if not missing:
        return circuit
    out = circuit.copy()
    for i in missing:
        out.append("DETECTOR", [stim.target_rec(i - n_meas)])
    return out

def merge_legs(circuit: stim.Circuit, legs: Sequence[int], n: int) -> tuple[stim.Circuit, tuple[int, ...]]:
    # fuse surplus CAT legs so that exactly n remain
    keep = list(legs[:n])
    drop = list(legs[n:])
    out = circuit.copy()
    for i, b in enumerate(drop):
        out.append("CNOT", [keep[i % n], b])
        out.append("M", [b])
        out.append("DETECTOR", [stim.target_rec(-1)])
    return out, tuple(keep)


def find_spidercat_root(hint: str | Path | None = None) -> Path:
    # locate the folder that contains circuits/ and circuits_data/ (SPIDERCAT GITHUB REPO MUST BE IN THE SAME FOLDER AS THIS)
    candidates: list[Path] = []
    if hint is not None:
        h = Path(hint)
        candidates += [h, h / "spidercat", h / "spidercat" / "spidercat"]
    here = Path.cwd()
    for base in [here, *here.parents][:4]:
        candidates += [base / "spidercat",  base / "spidercat-main" / "spidercat", base / "spidercat" / "spidercat"]
    for c in candidates:
        if (c / "circuits").is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "could not find the SpiderCat package folder containing circuits/ and circuits_data/")


class CatLibrary:
    # resolves (n, t) to a fault-tolerant CAT-state preparation circuit
    # resolution order:
    #   1. trivial cases n <= 3 
    #   2. circuits/cat_state_t{t}_n{n}_p{p}.stim shipped with / generated by SpiderCat
    #   3. circuits_data/*.json -> extract_circuit_rooted (needs the package importable)
    #   4. a fresh search via generate.cat_state_FT (slow & off by default)
    #   5. optional escalation to a larger t
    def __init__(self, root: str | Path | None = None, p: int = 1, allow_extract: bool = True, allow_generate: bool = False,
                 prefer: str = "disk",  escalate_t: bool = True, max_escalation: int = 4, pad_up: bool = True,
                 max_padding: int = 4, strict: bool = True):
        self.root = find_spidercat_root(root)
        self.p = p
        self.allow_extract = allow_extract
        if prefer not in ("disk", "extract"):
            raise ValueError("prefer must be 'disk' or 'extract'")
        self.prefer = prefer
        self.allow_generate = allow_generate
        self.escalate_t = escalate_t
        self.max_escalation = max_escalation
        self.pad_up = pad_up
        self.max_padding = max_padding
        self.strict = strict
        self._cache: dict[tuple[int, int], CatState] = {}

    def _stim_path(self, n: int, t: int) -> Path:
        return self.root / "circuits" / f"cat_state_t{t}_n{n}_p{self.p}.stim"

    def _json_path(self, n: int, t: int) -> Path:
        return self.root / "circuits_data" / f"cat_state_t{t}_n{n}_p{self.p}.json"

    def _from_disk(self, n: int, t: int) -> stim.Circuit | None:
        f = self._stim_path(n, t)
        return stim.Circuit(f.read_text()) if f.is_file() else None

    def _from_solution_triplet(self, n: int, t: int) -> stim.Circuit | None:
        # taken from spidercat demo
        # rebuild the circuit from the saved (graph, forest, marks) triplet
        if not self._json_path(n, t).is_file():
            return None
        try:
            import networkx as nx
            from spidercat.circuit_extraction import extract_circuit_rooted
            from spidercat.spanning_tree import (find_min_height_roots, match_forest_leaves_to_marked_edges)
        except ImportError:
            return None
        obj = json.loads(self._json_path(n, t).read_text())
        G = nx.from_edgelist(obj["G.edges"])
        M: dict[tuple[int, int], int] = {}
        for k, pairs in obj["M_inv"].items():
            for pair in pairs:
                M[tuple(pair)] = int(k)
        forest = nx.from_edgelist(obj["forest"])
        matchings = match_forest_leaves_to_marked_edges(G, forest, M)
        roots = find_min_height_roots(forest)
        return extract_circuit_rooted(G, forest, roots, M, matchings, verbose=False)

    # PRIVATE SO THAT THE METHODS DONT CLASH WITH SPIDERCAT mainly
    # need to test if they doo later
    def _generate(self, n: int, t: int) -> stim.Circuit | None:
        try:
            from spidercat.generate import cat_state_FT
        except ImportError:
            return None
        circs = cat_state_FT(n, t, (self.p,), run_verification=False, replace=False)
        return circs.get(self.p)

    @staticmethod
    def _small(n: int) -> stim.Circuit | None:
        if n < 1:
            raise ValueError("cat states need n >= 1")
        if n == 1:
            c = stim.Circuit()
            c.append("H", 0)
            return c
        if n <= 3:
            c = stim.Circuit()
            c.append("H", 0)
            for i in range(1, n):
                c.append("CNOT", [0, i])
            return c
        return None

    def _one_flagged(self, n: int) -> stim.Circuit:
        c = stim.Circuit()
        c.append("H", 0)
        c.append("CNOT", [0, n])
        for i in range(1, n - 1, 2):
            c.append("CNOT", [0, i])
            c.append("CNOT", [n, i + 1])
        if n % 2 == 0:
            c.append("CNOT", [0, n - 1])
        c.append("CNOT", [0, n])
        c.append("M", n)
        c.append("DETECTOR", [stim.target_rec(-1)])
        return c

    # public
    def cat_circuit(self, n: int, t: int) -> CatState:
        # best available CAT-state prep with n legs, tolerating t faults
        key = (n, t)
        if key in self._cache:
            return self._cache[key]

        t_eff = min(t, t_cap(n))

        # candidate (builder, achieved-t, description)
        cands: list = []

        if n <= 3:
            cands.append((lambda: self._small(n), n, t_eff, f"unflagged cat (n={n})"))
        else:
            if self.prefer == "extract" and self.allow_extract:
                for label in dict.fromkeys([t, t_eff]):
                    cands.append((lambda l = label: self._from_solution_triplet(n, l), n, t_eff,
                                  f"extract_circuit_rooted from circuits_data/cat_state_t{label}_n{n}"))
            # the shipped files are labelled by the requested t even though the circuit only realises min(t, floor(n/2)-1)
            # so try both labels?
            for label in dict.fromkeys([t, t_eff]):
                cands.append((lambda l = label: self._from_disk(n, l), n, t_eff, f"circuits/cat_state_t{label}_n{n}_p{self.p}.stim"))
            if self.escalate_t:
                for k in range(1, self.max_escalation + 1):
                    cands.append((lambda l=t + k: self._from_disk(n, l), n, t_eff, f"circuits/cat_state_t{t + k}_n{n}_p{self.p}.stim (escalated label)"))
            if n <= 5 or t_eff <= 1:
                cands.append((lambda: self._one_flagged(n), n, min(t_eff, 1), f"one_flagged cat (n={n})"))
            if self.allow_extract:
                for label in dict.fromkeys([t, t_eff]):
                    cands.append((lambda l = label: self._from_solution_triplet(n, l), n, t_eff, f"extract_circuit_rooted from circuits_data/cat_state_t{label}_n{n}"))
            if self.allow_generate:
                cands.append((lambda: self._generate(n, t), n, t_eff, "generate.cat_state_FT (fresh graph search)"))
            if self.pad_up:
                for extra in range(1, self.max_padding + 1):
                    for label in dict.fromkeys([t, min(t, t_cap(n + extra))]):
                        cands.append((lambda m = n + extra, l=label: self._from_disk(m, l),
                                      n + extra, min(t, t_cap(n + extra)),
                                      f"circuits/cat_state_t{label}_n{n + extra}_p{self.p}.stim "
                                      f"-> {extra} leg(s) merged"))

        for build, n_built, t_achieved, src in cands:
            circuit = build()
            if circuit is None:
                continue
            cat = self._finalise(circuit, n, t_achieved, t, src)
            if cat is not None:
                self._cache[key] = cat
                return cat

        msg = (
            f"no fault-tolerant CAT state found for n={n}, t={t} (p={self.p})"
            f"the best achievable here is t={t_eff} looking in {self.root / 'circuits'} "
            f"options: allow_generate=True to search for a new graph, raise max_padding, "
            f"lower t, or pre-generate with python generate.py -n {n} -t {t} -o {self.root}")
        if self.strict:
            raise LookupError(msg)
        cat = self._finalise(self._unflagged(n), n, 0, t, "NOT FAULT TOLERANT: " + msg)
        assert cat is not None
        self._cache[key] = cat
        return cat

    @staticmethod
    def _unflagged(n: int) -> stim.Circuit:
        c = stim.Circuit()
        c.append("H", 0)
        for i in range(1, n):
            c.append("CNOT", [0, i])
        return c

    def _finalise(self, circuit: stim.Circuit, n: int, t: int, t_requested: int, source: str) -> CatState | None:
        circuit = postselect_uncovered_measurements(circuit)
        measured = measured_qubits(circuit)
        legs = tuple(q for q in range(circuit.num_qubits) if q not in measured)
        if len(legs) < n:
            return None
        if len(legs) > n:
            circuit, legs = merge_legs(circuit, legs, n)
        return CatState(circuit = circuit,  n = n, t = t, t_requested = t_requested, legs = legs,
                        num_flag_measurements = circuit.num_measurements, source = source)

    def marked_graph(self, n: int, t: int):
        # the 3-regular marked graph behind the (n, t) cat, as (G, M, t, source)
        import networkx as nx

        t_eff = min(t, t_cap(n))
        labels = list(dict.fromkeys([t, t_eff] + [t + k for k in range(1, self.max_escalation + 1)]))
        for label in labels:
            f = self._json_path(n, label)
            if not f.is_file():
                continue
            obj = json.loads(f.read_text())
            G = nx.from_edgelist(obj["G.edges"])
            M: dict[tuple[int, int], int] = {}
            for k, pairs in obj["M_inv"].items():
                for pair in pairs:
                    u, v = sorted(pair)
                    M[(u, v)] = int(k)
            if sum(M.values()) != n:
                continue
            return G, M, t_eff, f"circuits_data/{f.name}"
        return None

    def verify(self, cat: CatState) -> bool:
        # simulate the prep, post-select the flags, and check the state is |CAT_n>
        sim = stim.TableauSimulator()
        for op in cat.circuit.flattened():
            if op.name == "DETECTOR":
                continue
            if op.name in ("M", "MZ"):
                for tg in op.targets_copy():
                    sim.postselect_z(tg.qubit_value, desired_value=False)
            elif op.name == "MX":
                for tg in op.targets_copy():
                    sim.postselect_x(tg.qubit_value, desired_value=False)
            else:
                sim.do(op)
        nq = cat.circuit.num_qubits
        x_all = ["_"] * nq
        for q in cat.legs:
            x_all[q] = "X"
        if sim.peek_observable_expectation(stim.PauliString("".join(x_all))) != 1:
            return False
        for a, b in zip(cat.legs, cat.legs[1:]):
            zz = ["_"] * nq
            zz[a] = zz[b] = "Z"
            if sim.peek_observable_expectation(stim.PauliString("".join(zz))) != 1:
                return False
        return True

# ---------------------------------------------------------------------------
# shor gadget
# ---------------------------------------------------------------------------
def offset_circuit(circ: stim.Circuit, offset: int) -> stim.Circuit:
    # shift every qubit target by offset (rec[] targets are relative, so untouched)
    out = stim.Circuit()
    for op in circ.flattened():
        new_targets = []
        for t in op.targets_copy():
            if t.is_qubit_target:
                new_targets.append(stim.GateTarget(t.value + offset))
            elif t.is_x_target:
                new_targets.append(stim.target_x(t.value + offset))
            elif t.is_y_target:
                new_targets.append(stim.target_y(t.value + offset))
            elif t.is_z_target:
                new_targets.append(stim.target_z(t.value + offset))
            else:
                new_targets.append(t)
        out.append(stim.CircuitInstruction(op.name, new_targets, op.gate_args_copy()))
    return out

def shor_gadget(cat: CatState, terms: Sequence[PauliTerm],  ancilla_base: int, reset: bool = True, 
                tick: bool = True) -> tuple[stim.Circuit, list[int]]:
    # build the cat-state measurement of prod(terms)

    # returns (circuit, syndrome_offsets) where syndrome_offsets are negative red offsets 
    # relative to the end of the returned circui whose XOR is the measured eigenvalue (0 = +1)
    if len(terms) != cat.n:
        raise ValueError(f"cat has {cat.n} legs but the product has weight {len(terms)}")

    circ = stim.Circuit()
    prep = offset_circuit(cat.circuit, ancilla_base)
    if reset:
        circ.append("R", [ancilla_base + q for q in range(cat.num_qubits)])
    circ += prep
    if tick:
        circ.append("TICK")

    # leg i of the cat drives term i of the product
    leg_q = [ancilla_base + q for q in cat.legs]
    by_gate: dict[str, list[int]] = {"CX": [], "CY": [], "CZ": []}
    for leg, term in zip(leg_q, terms):
        by_gate[CONTROLLED[term.pauli]] += [leg, term.qubit]
    for gate in ("CX", "CY", "CZ"):
        if by_gate[gate]:
            circ.append(gate, by_gate[gate])

    # an odd number of inverted targets flips the expected parity
    if sum(1 for t in terms if t.inverted) % 2:
        circ.append("Z", [leg_q[0]])

    if tick:
        circ.append("TICK")
    circ.append("MX", leg_q)
    syndrome_offsets = list(range(-cat.n, 0))
    return circ, syndrome_offsets


# whole-circuit rewriting
_NAME_RE = re.compile(r"(?:^|[^a-z])n[_-]?(\d+).*?k[_-]?(\d+).*?d[_-]?(\d+)", re.I)
def code_params_from_name(name: str) -> dict[str, int]:
    """Pull ``n``, ``k``, ``d`` out of names like ``q4_n_25_k_4_d_7syndrome_mpp.stim``."""
    m = _NAME_RE.search(Path(name).stem)
    if not m:
        return {}
    return {"n": int(m.group(1)), "k": int(m.group(2)), "d": int(m.group(3))}


@dataclass
class GadgetInfo:
    index: int
    terms: list[PauliTerm]
    weight: int
    cat: CatState | None
    syndrome_recs: list[int] = field(default_factory = list)   
    flag_recs: list[int] = field(default_factory = list)


@dataclass
class RewriteResult:
    circuit: stim.Circuit
    gadgets: list[GadgetInfo]
    num_data_qubits: int
    num_qubits: int
    source: Path | None = None

    def summary(self) -> str:
        rows = [f"{'#':>3} {'weight':>6} {'t':>3} {'flags':>5} {'ancillas':>8}  source", "-" * 78]
        for g in self.gadgets:
            if g.cat is None:
                rows.append(f"{g.index:>3} {g.weight:>6} {'-':>3} {'-':>5} {'-':>8}  passed through")
            else:
                rows.append(f"{g.index:>3} {g.weight:>6} {g.cat.t:>3} {g.cat.num_flags:>5} {g.cat.num_qubits:>8} {g.cat.source}")
        rows.append("-" * 78)
        rows.append(f"data qubits: {self.num_data_qubits} total qubits: {self.num_qubits} gadgets: {sum(1 for g in self.gadgets if g.cat)}")
        return "\n".join(rows)

    def to_json(self) -> dict:
        return {"source": str(self.source) if self.source else None, "num_data_qubits": self.num_data_qubits, 
        "num_qubits": self.num_qubits, "gadgets": [ {
             "index": g.index, "pauli": pauli_product_str(g.terms), "weight": g.weight,
            "t": g.cat.t if g.cat else None, "t_requested": g.cat.t_requested if g.cat else None,
            "num_flags": g.cat.num_flags if g.cat else None, "cat_source": g.cat.source if g.cat else None,
            "syndrome_measurement_indices": g.syndrome_recs, "flag_measurement_indices": g.flag_recs} for g in self.gadgets]}

class MppCatRewriter:
    def __init__(
        self,
        library: CatLibrary,
        t: int | None = None,
        distance: int | None = None,
        t_rule=lambda d: d // 2,
        min_weight: int = 2,
        ancilla_mode: str = "reuse",
        reset: bool = True,
        syndrome: str = "observable",
        keep_original_mpp: bool = False):
        
        if t is None and distance is None:
            raise ValueError("pls give either t or distance")
        self.library = library
        self.t = int(t) if t is not None else int(t_rule(distance))
        self.distance = distance
        self.min_weight = min_weight
        if ancilla_mode not in ("reuse", "fresh"):
            raise ValueError("ancilla_mode")
        self.ancilla_mode = ancilla_mode
        self.reset = reset
        if syndrome not in ("none", "observable", "detector"):
            raise ValueError("syndrome")
        self.syndrome = syndrome
        self.keep_original_mpp = keep_original_mpp

    def rewrite(self, circuit: stim.Circuit | str, source: Path | None = None) -> RewriteResult:
        if isinstance(circuit, str):
            circuit = stim.Circuit(circuit)
        flat = circuit.flattened()
        n_data = flat.num_qubits

        out = stim.Circuit()
        gadgets: list[GadgetInfo] = []
        rec_map: list[list[int]] = []
        n_old_meas = 0
        n_new_meas = 0
        next_free = n_data
        max_qubit = n_data - 1
        gadget_index = 0
        obs_index = 0

        for op in flat:
            name = op.name

            if name == "MPP":
                for terms in parse_pauli_products(op.targets_copy()):
                    weight = len(terms)
                    if weight < self.min_weight:
                        # too small to be worth a cat: emit the original MPP untouched
                        sub = stim.Circuit()
                        sub.append("MPP",  _mpp_targets(terms))
                        out += sub
                        rec_map.append([n_new_meas])
                        n_new_meas += 1
                        n_old_meas += 1
                        gadgets.append(GadgetInfo(gadget_index, list(terms), weight, None))
                        gadget_index += 1
                        continue

                    cat = self.library.cat_circuit(weight, self.t)
                    base = next_free if self.ancilla_mode == "fresh" else n_data
                    gadget, syn_off = shor_gadget(cat, terms, base, reset=(self.reset or self.ancilla_mode == "reuse"))
                    out += gadget
                    n_here = gadget.num_measurements
                    flag_recs = [n_new_meas + i for i in range(n_here - cat.n)]
                    syn_recs = [n_new_meas + n_here + o for o in syn_off]
                    n_new_meas += n_here
                    rec_map.append(syn_recs)
                    n_old_meas += 1

                    if self.syndrome == "observable":
                        out.append("OBSERVABLE_INCLUDE", [stim.target_rec(o) for o in syn_off],  obs_index)
                        obs_index += 1
                    elif self.syndrome == "detector":
                        out.append("DETECTOR", [stim.target_rec(o) for o in syn_off])

                    if self.keep_original_mpp:
                        out.append("MPP", _mpp_targets(terms))
                        n_new_meas += 1

                    max_qubit = max(max_qubit, base + cat.num_qubits - 1)
                    if self.ancilla_mode == "fresh":
                        next_free = base + cat.num_qubits

                    gadgets.append(GadgetInfo(gadget_index, list(terms), weight, cat, syn_recs, flag_recs))
                    gadget_index += 1
                continue

            if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                new_targets = []
                for t in op.targets_copy():
                    if t.is_measurement_record_target:
                        old_idx = n_old_meas + t.value
                        for new_idx in rec_map[old_idx]:
                            new_targets.append(stim.target_rec(new_idx - n_new_meas))
                    else:
                        new_targets.append(t)
                out.append(stim.CircuitInstruction(name, new_targets, op.gate_args_copy()))
                continue

            out.append(op)
            if name in MEASUREMENT_GATES:
                k = sum(1 for t in op.targets_copy() if t.is_qubit_target)
                for _ in range(k):
                    rec_map.append([n_new_meas])
                    n_new_meas += 1
                    n_old_meas += 1
            qs = [t.qubit_value for t in op.targets_copy() if t.is_qubit_target]
            if qs:
                max_qubit = max(max_qubit, max(qs))

        return RewriteResult(out, gadgets, n_data, max_qubit + 1, source)

    def rewrite_file(self, path: str | Path) -> RewriteResult:
        path = Path(path)
        params = code_params_from_name(path.name)
        # t was fixed explicitly; leave it
        if self.distance is None and "d" in params:
            pass  
        return self.rewrite(stim.Circuit(path.read_text()), source=path)

    def rewrite_folder(self, in_dir: str | Path, out_dir: str | Path, pattern: str = "*.stim", 
                       write_json: bool = True, per_file_distance: bool = True) -> dict[str, RewriteResult]:
        in_dir, out_dir = Path(in_dir), Path(out_dir)
        out_dir.mkdir(parents = True, exist_ok = True)
        results: dict[str, RewriteResult] = {}
        base_t = self.t
        for f in sorted(in_dir.glob(pattern)):
            params = code_params_from_name(f.name)
            if per_file_distance and "d" in params:
                self.t = params["d"] // 2
            try:
                res = self.rewrite_file(f)
            finally:
                self.t = base_t
            res.circuit.to_file(str(out_dir / f.name))
            if write_json:
                (out_dir / (f.stem + ".json")).write_text(json.dumps(res.to_json(), indent=2))
            results[f.name] = res
        return results


def _mpp_targets(terms: Iterable[PauliTerm]) -> list[stim.GateTarget]:
    maker = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
    out: list[stim.GateTarget] = []
    for i, term in enumerate(terms):
        if i:
            out.append(stim.target_combiner())
        out.append(maker[term.pauli](term.qubit, term.inverted))
    return out


# CLI
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog = "spidercat-shor", description = "Replace high-weight MPPs with SpiderCat fault-tolerant cat-state gadgets.")
    ap.add_argument("inputs", nargs = "+", type = Path, help = ".stim files or folders")
    ap.add_argument("-o", "--out-dir", type = Path, default = Path("circuits_cat"))
    ap.add_argument("-s", "--spidercat", type = Path, default = None, help = "folder containing circuits/ and circuits_data/")
    ap.add_argument("-t", "--fault-distance", type = int, default = None, help = "cat fault tolerance; default d//2 from the filename")
    ap.add_argument("-d", "--distance", type = int, default = None)
    ap.add_argument("--min-weight", type = int, default = 2)
    ap.add_argument("--ancillas", choices = ["reuse", "fresh"], default = "reuse")
    ap.add_argument("--syndrome", choices = ["none", "observable", "detector"], default = "observable")
    ap.add_argument("--generate", action = "store_true", help = "allow searching for a new graph when (n,t) is not cached")
    ap.add_argument("--no-escalate", action = "store_true", help = "fail instead of falling back to a larger t")
    args = ap.parse_args(argv)

    lib = CatLibrary(args.spidercat, allow_generate = args.generate, escalate_t = not args.no_escalate)
    rw = MppCatRewriter(lib, t = args.fault_distance, distance = args.distance if args.fault_distance is None else None,
                        min_weight = args.min_weight, ancilla_mode = args.ancillas, 
                        syndrome = args.syndrome) if (args.fault_distance or args.distance) else MppCatRewriter(lib, t = 1)

    files: list[Path] = []
    for p in args.inputs:
        files += sorted(p.glob("*.stim")) if p.is_dir() else [p]

    args.out_dir.mkdir(parents = True, exist_ok = True)
    for f in files:
        params = code_params_from_name(f.name)
        if args.fault_distance is None and "d" in params:
            rw.t = params["d"] // 2
        res = rw.rewrite_file(f)
        res.circuit.to_file(str(args.out_dir / f.name))
        (args.out_dir / (f.stem + ".json")).write_text(json.dumps(res.to_json(), indent=2))
        print(f"\n=== {f.name}  (t={rw.t}) ===")
        print(res.summary())
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



def atomise(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    # split fused instructions into single gates.
    # stim merges consecutive identical instructions, so CX 0 1 then CX 0 2 becomes one CX 0 1 0 2
    atoms: list[stim.CircuitInstruction] = []
    for op in circuit.flattened():
        if op.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "SHIFT_COORDS", "QUBIT_COORDS"):
            continue
        targets = op.targets_copy()
        step = 2 if op.name in _TWO_QUBIT_GATES else 1
        args = op.gate_args_copy()
        for i in range(0, len(targets), step):
            atoms.append(stim.CircuitInstruction(op.name, targets[i:i + step], args))
    return atoms


# optional: brute force
def verify_fault_tolerance(cat: CatState, max_weight: int = 2, max_samples: int | None = 200_000, seed: int = 0) -> dict:  
    ops = atomise(cat.circuit)
    locations: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        for tg in op.targets_copy():
            if tg.is_qubit_target:
                locations.append((i, tg.qubit_value))
    faults = [(i, q, p) for (i, q) in locations for p in "XYZ"]

    n, legs, nq = cat.n, cat.legs, cat.circuit.num_qubits
    rng = random.Random(seed)
    worst = 0
    witness = None
    rejected = tested = 0

    def run(combo) -> int | None:
        by_index: dict[int, list[tuple[int, str]]] = {}
        for (i, q, p) in combo:
            by_index.setdefault(i, []).append((q, p))
        sim = stim.TableauSimulator()
        for i, op in enumerate(ops):
            if op.name in ("M", "MZ"):
                for tg in op.targets_copy():
                    try:
                        sim.postselect_z(tg.qubit_value, desired_value=False)
                    except ValueError:
                        return None
            elif op.name == "MX":
                for tg in op.targets_copy():
                    try:
                        sim.postselect_x(tg.qubit_value, desired_value=False)
                    except ValueError:
                        return None
            else:
                sim.do(op)
            for (q, p) in by_index.get(i, ()):
                sim.do(stim.Circuit(f"{p} {q}"))

        e = [0]
        for a, b in zip(legs, legs[1:]):
            s = ["_"] * nq
            s[a] = s[b] = "Z"
            val = sim.peek_observable_expectation(stim.PauliString("".join(s)))
            if val == 0:
                # not a CAT state at all -> maximal failure
                return n           
            e.append(e[-1] ^ (1 if val == -1 else 0))
        w = sum(e)
        return min(w, n - w)

    for weight in range(1, max_weight + 1):
        combos = itertools.combinations(faults, weight)
        total = 1
        for k in range(weight):
            total = total * (len(faults) - k) // (k + 1)
        if max_samples is not None and total > max_samples:
            combos = (tuple(rng.sample(faults, weight)) for _ in range(max_samples))
        for combo in combos:
            tested += 1
            out = run(combo)
            if out is None:
                rejected += 1
                continue
            if out > weight and out > worst:
                worst, witness = out, combo
    return {"ok": witness is None, "n": n, "t_claimed": cat.t, "max_weight_tested": max_weight,
            "num_fault_locations": len(locations),  "faults_tested": tested, "rejected_by_flags": rejected,
            "worst_output_weight": worst, "witness": witness, "source": cat.source}

# circuit depth
def circuit_depth(circuit: stim.Circuit, mode: str = "two_qubit") -> int:
    if mode not in ("two_qubit", "all", "gates"):
        raise ValueError("mode must be 'two_qubit', 'all' or 'gates'")

    def costs(name: str) -> bool:
        if mode == "two_qubit":
            return name in _ENTANGLING
        if mode == "gates":
            return name not in MEASUREMENT_GATES and name not in ("R", "RX", "RY", "RZ")
        return True

    ready: dict[int, int] = {}
    for op in atomise(circuit):
        qs = [t.qubit_value for t in op.targets_copy() if t.is_qubit_target]
        if not qs:
            continue
        start = max((ready.get(q, 0) for q in qs), default = 0)
        end = start + (1 if costs(op.name) else 0)
        for q in qs:
            ready[q] = end
    return max(ready.values(), default = 0)

def depth_report(result: "RewriteResult") -> dict:
    # depth of a rewritten circuit whole and per gadget
    whole = {m: circuit_depth(result.circuit, m) for m in ("two_qubit", "gates", "all")}
    per_gadget = []
    for g in result.gadgets:
        if g.cat is None:
            per_gadget.append({"index": g.index, "weight": g.weight, "cat_depth": None, "gadget_depth": None})
            continue
        prep = circuit_depth(g.cat.circuit, "two_qubit")
        # + 1 layer of controlled-Paulis (all act on disjoint pairs), + the MX layer
        per_gadget.append({"index": g.index, "weight": g.weight, "t": g.cat.t,
                           "cat_prep_cnot_depth": prep, "gadget_cnot_depth": prep + 1})
    return {"whole_circuit": whole, "per_gadget": per_gadget,
            "sum_of_gadget_depths": sum(d["gadget_cnot_depth"] for d in per_gadget if d.get("gadget_cnot_depth"))}

def atomise_all(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    atoms: list[stim.CircuitInstruction] = []
    for op in circuit.flattened():
        if op.name == "TICK":
            continue
        if op.name in _ANNOTATIONS or op.name == "QUBIT_COORDS":
            atoms.append(op)
            continue
        targets = op.targets_copy()
        step = 2 if op.name in _TWO_QUBIT_GATES else 1
        args = op.gate_args_copy()
        for i in range(0, len(targets), step):
            atoms.append(stim.CircuitInstruction(op.name, targets[i:i + step], args))
    return atoms

def schedule_ticks(circuit: stim.Circuit) -> stim.Circuit:
    # rewrite a circuit into explicit parallel layers separated by TICK
    layers: dict[int, list[stim.CircuitInstruction]] = defaultdict(list)
    ready: dict[int, int] = {}
    last_meas_layer = 0

    for op in atomise_all(circuit):
        name = op.name
        if name == "QUBIT_COORDS":
            layers[0].append(op)
            continue
        if name in _ANNOTATIONS:
            layers[last_meas_layer].append(op)
            continue
        qs = [t.qubit_value for t in op.targets_copy() if t.is_qubit_target]
        if not qs:
            layers[last_meas_layer].append(op)
            continue
        start = max((ready.get(q, 0) for q in qs), default = 0)
        if name in MEASUREMENT_GATES:
            start = max(start, last_meas_layer)
            last_meas_layer = start
        layers[start].append(op)
        for q in qs:
            ready[q] = start + 1

    out = stim.Circuit()
    for i in sorted(layers):
        if i and len(out):
            out.append("TICK")
        for op in layers[i]:
            out.append(op)
    return out


def check_schedule(original: stim.Circuit, scheduled: stim.Circuit) -> dict:
    same = {"num_qubits": original.num_qubits == scheduled.num_qubits,
            "num_measurements": original.num_measurements == scheduled.num_measurements,
            "num_detectors": original.num_detectors == scheduled.num_detectors,
            "num_observables": original.num_observables == scheduled.num_observables}
    fired = {}
    for label, c in (("original", original), ("scheduled", scheduled)):
        try:
            fired[label] = bool(c.compile_detector_sampler().sample(256).any())
        except Exception as exc:
            fired[label] = f"error: {exc}"
    same["detectors_quiet"] = fired
    same["depth"] = {"ticks_in_scheduled": scheduled.num_ticks,  "asap_all": circuit_depth(original, "all"), 
                     "asap_two_qubit": circuit_depth(original, "two_qubit")}
    same["ok"] = all(v is True for k, v in same.items() if isinstance(v, bool)) and \
        fired["original"] == fired["scheduled"] is False
    return same