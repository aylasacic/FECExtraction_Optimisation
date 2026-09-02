from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

from CircuitExtraction.circuit_extractor import CircuitExtractor

from .layers import components
from .primitives import vtype_name

__all__ = ["EdgeRef", "ScheduleExtractionResult", "extract_schedule"]


@dataclass(frozen = True)
class EdgeRef:
    source: Hashable
    target: Hashable
    key: Hashable
    data: Mapping[str, Any]
    @property
    def uid(self) -> Tuple[Hashable, Hashable, Hashable]:
        return (self.source, self.target, self.key)


@dataclass
class ScheduleExtractionResult:
    circuit: Any
    extractor: Any
    input_edges: List[EdgeRef]
    output_edges: List[EdgeRef]
    input_edge_to_qubit: Dict[Tuple[Hashable, Hashable, Hashable], int]
    output_edge_to_qubit: Dict[Tuple[Hashable, Hashable, Hashable], int]
    output_qubits: List[int]
    ignored_scalars: List[Tuple[Tuple[Hashable, ...], str, Any]]


def _iter_edges(G):
    if G.is_multigraph():
        for u, v, key, data in G.edges(keys=True, data=True):
            yield u, v, key, dict(data)
    else:
        for index, (u, v, data) in enumerate(G.edges(data=True)):
            yield u, v, index, dict(data)


def _boundary_index(data):
    for name in ("boundary_qubit", "qubit", "wire", "index"):
        value = data.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _edge_sort_key(edge, node_position):
    boundary = _boundary_index(edge.data)
    return (boundary is None,
            boundary if boundary is not None else 0,
            node_position.get(edge.source, 10 ** 12),
            node_position.get(edge.target, 10 ** 12),
            repr(edge.source), repr(edge.target), repr(edge.key))


def _spider_type(G, nodes):
    kinds = []
    for node in nodes:
        raw = G.nodes[node].get("vtype")
        kind = vtype_name(raw)
        if kind is None:
            raise ValueError(f"vertex {node!r} has unsupported vtype {raw!r}; expected Z or X")
        kinds.append(kind)
    if len(set(kinds)) != 1:
        raise ValueError(f"same-layer component mixes spider colours: {list(zip(nodes, kinds))!r}")
    return kinds[0]


def _phase_sum(G, nodes):
    phase = Fraction(0, 1)
    for node in nodes:
        phase += G.nodes[node].get("phase", 0)
    try:
        phase %= 2
    except TypeError:
        pass
    return phase


def _is_hadamard(data):
    raw = data.get("etype", data.get("edge_type", data.get("type")))
    if raw is None:
        return False
    name = getattr(raw, "name", str(raw)).upper()
    return name in {"H", "HAD", "HADAMARD", "2"} or name.endswith(".HADAMARD")


def _execute_spider(extractor, incoming, n_outputs, spider_type, phase):
    """One generalized spider, preserving merge -> phase -> split order."""
    if not incoming:
        if n_outputs == 0:
            return []
        return list(extractor.split_zero_to_many(n_outputs, spider_type, phase=phase))

    pivot = incoming[0]
    if len(incoming) > 1:
        pivot = extractor.merge_many_to_one(list(incoming), spider_type)

    extractor.apply_spider(pivot, spider_type, phase)

    if n_outputs == 0:
        # Z effect is <+|, X effect is <0|, matching split_zero_to_manys
        # Z preparation |+> and X preparation |0>
        extractor.postselect(pivot, "+" if spider_type == "Z" else "0")
        extractor.history.append(("cat_effect", tuple(incoming), spider_type, phase))
        return []
    if n_outputs == 1:
        return [pivot]
    return list(extractor.split_one_to_many(pivot, n_outputs, spider_type))


def extract_schedule(G, schedule, *, extractor=None, extractor_cls=None,
                     input_node="I", output_node="O",
                     fuse_same_layer_spiders=False, strict_scalars=True):
    layers = [list(layer) for layer in schedule if layer]
    flat = [node for layer in layers for node in layer]
    if len(flat) != len(set(flat)):
        raise ValueError("every scheduled vertex must appear exactly once")

    internal_nodes = set(G.nodes()) - {input_node, output_node}
    missing = internal_nodes - set(flat)
    extra = set(flat) - internal_nodes
    if missing or extra:
        raise ValueError(f"schedule/graph mismatch: missing={sorted(map(repr, missing))}, extra = {sorted(map(repr, extra))}")

    node_to_layer = {node: t for t, layer in enumerate(layers) for node in layer}
    node_position = {node: i for i, node in enumerate(flat)}

    def time_of(node):
        if node == input_node:
            return -1
        if node == output_node:
            return len(layers)
        return node_to_layer[node]

    oriented, same_layer_edges = [], []
    for u, v, key, data in _iter_edges(G):
        tu, tv = time_of(u), time_of(v)
        if tu == tv:
            same_layer_edges.append((u, v, key, data))
        elif tu < tv:
            oriented.append(EdgeRef(u, v, key, data))
        else:
            oriented.append(EdgeRef(v, u, key, data))

    if same_layer_edges and not fuse_same_layer_spiders:
        preview = [(u, v, key) for u, v, key, _ in same_layer_edges[:8]]
        raise ValueError("schedule contains same-layer graph edges. Ordinary extraction has no causal "
            f"orientation for them: {preview!r}. Keep ALLOW_CO_MEASURE=False [TO BE FIXED]")

    if fuse_same_layer_spiders:
        for u, v, key, data in same_layer_edges:
            if _is_hadamard(data):
                raise ValueError(f"cannot fuse same-layer Hadamard edge {(u, v, key)!r}")
            if _spider_type(G, (u,)) != _spider_type(G, (v,)):
                raise ValueError(f"cannot fuse differently coloured same-layer edge {(u, v, key)!r}")

    incoming_by_node = {node: [] for node in flat}
    outgoing_by_node = {node: [] for node in flat}
    for edge in oriented:
        if edge.target in incoming_by_node:
            incoming_by_node[edge.target].append(edge)
        if edge.source in outgoing_by_node:
            outgoing_by_node[edge.source].append(edge)

    input_edges = sorted([e for e in oriented if e.source == input_node], key = lambda e: _edge_sort_key(e, node_position))
    output_edges = sorted([e for e in oriented if e.target == output_node], key = lambda e: _edge_sort_key(e, node_position))
    boundary_ids = [_boundary_index(e.data) for e in input_edges]
    use_boundary_ids = (all(i is not None for i in boundary_ids) and len(set(boundary_ids)) == len(boundary_ids)
                        and set(boundary_ids) == set(range(len(input_edges))))
    input_labels = ([int(i) for i in boundary_ids] if use_boundary_ids else list(range(len(input_edges))))

    if extractor is None:
        if extractor_cls is None:
            extractor_cls = CircuitExtractor
        extractor = extractor_cls(len(input_edges))

    live, input_edge_to_qubit = {}, {}
    for edge, qubit in zip(input_edges, input_labels):
        live[edge.uid] = qubit
        input_edge_to_qubit[edge.uid] = qubit

    ignored_scalars = []

    for t, layer in enumerate(layers):
        if fuse_same_layer_spiders:
            groups = components(layer, [(u, v) for u, v, _, _ in same_layer_edges])
        else:
            groups = [(node,) for node in layer]

        for component in groups:
            members = set(component)
            incoming = sorted([e for node in component for e in incoming_by_node[node] if e.source not in members],
                              key=lambda e: _edge_sort_key(e, node_position))
            outgoing = sorted([e for node in component for e in outgoing_by_node[node] if e.target not in members],
                              key=lambda e: _edge_sort_key(e, node_position))

            input_qubits = []
            for edge in incoming:
                if edge.uid not in live:
                    raise RuntimeError(f"edge {edge.uid!r} is not live when layer {t} consumes it")
                qubit = live.pop(edge.uid)
                if _is_hadamard(edge.data):
                    extractor.add_gate("HAD", qubit)
                input_qubits.append(qubit)

            spider_type = _spider_type(G, component)
            phase = _phase_sum(G, component)

            if not input_qubits and not outgoing:
                if strict_scalars:
                    raise ValueError(f"component {component!r} is a 0->0 scalar and "
                                     "cannot be represented by a circuit wire")
                ignored_scalars.append((component, spider_type, phase))
                continue

            output_qubits = _execute_spider(extractor, input_qubits, len(outgoing),
                                            spider_type, phase)
            if len(output_qubits) != len(outgoing):
                raise RuntimeError(f"extractor returned {len(output_qubits)} outputs for "
                                   f"component {component!r}, but graph requires {len(outgoing)}")
            for edge, qubit in zip(outgoing, output_qubits):
                if edge.uid in live:
                    raise RuntimeError(f"edge {edge.uid!r} was assigned twice")
                live[edge.uid] = qubit

        extractor.next_step()

    output_edge_to_qubit, output_qubits = {}, []
    for edge in output_edges:
        if edge.uid not in live:
            raise RuntimeError(f"output edge {edge.uid!r} has no live qubit")
        qubit = live.pop(edge.uid)
        if _is_hadamard(edge.data):
            extractor.add_gate("HAD", qubit)
        output_edge_to_qubit[edge.uid] = qubit
        output_qubits.append(qubit)

    if live:
        raise RuntimeError(f"extraction left non-output live edges: {list(live)!r}")

    return ScheduleExtractionResult(
        circuit = extractor.get_circuit(), extractor = extractor,
        input_edges = input_edges, output_edges = output_edges,
        input_edge_to_qubit = input_edge_to_qubit,
        output_edge_to_qubit = output_edge_to_qubit,
        output_qubits = output_qubits, ignored_scalars = ignored_scalars)
