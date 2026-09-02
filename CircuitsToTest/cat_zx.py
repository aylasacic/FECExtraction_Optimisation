from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path
import stim
import numpy as np
import pyzx as zx
from pyzx.utils import EdgeType, VertexType

from spidercat_shor import CatLibrary, t_cap

_DEF_KEYWORDS = ("import ", "from ", "def ", "class ", "@", "#")
__all__ = ["StimToPyZXExtractor", "stim_to_pyzx", "mpp_to_cat_zx", "use_extractor"]


def _load_extractor(path: str | Path | None = None):
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        here = Path(__file__).resolve().parent
        for d in (here, here.parent, Path.cwd()):
            candidates += sorted(d.glob("StimToPyZX*.py"))
    for c in candidates:
        if not c.is_file():
            continue
        lines = c.read_text().splitlines(keepends=True)
        kept = []
        for line in lines:
            stripped = line.strip()
            if (stripped and not line[0].isspace() and not any(stripped.startswith(k) for k in _DEF_KEYWORDS)
                and not stripped.startswith(("'", '"', ")", "]", "}"))):
                break
            kept.append(line)
        ns: dict = {"__name__": "stim_to_pyzx_module", "__file__": str(c)}
        exec(compile("".join(kept), str(c), "exec"), ns)
        if "StimToPyZXExtractor" in ns:
            return ns
    raise ImportError("BAD: StimToPyZX_*.py not in folder. Put it next to cat_zx.py, or call cat_zx.use_extractor('/path/to/StimToPyZX*.py').")


_NS = _load_extractor()
StimToPyZXExtractor = _NS["StimToPyZXExtractor"]
stim_to_pyzx = _NS["stim_to_pyzx"]


def use_extractor(path):
    global _NS, StimToPyZXExtractor, stim_to_pyzx
    _NS = _load_extractor(path)
    StimToPyZXExtractor = _NS["StimToPyZXExtractor"]
    stim_to_pyzx = _NS["stim_to_pyzx"]
    # CatMppExtractor.__bases__ = (StimToPyZXExtractor,)

# I messed something up in the other import, FIX LATER
def stim_to_pyzx(
    circuit,
    cat_library: CatLibrary | None = None,
    t: int | None = None,
    spidercat_hub: bool = False,
    min_weight: int = 4,
    open_inputs: bool = True,
    postselect: bool = True,
    init=None,
    prep_color: str = "match",
    serialize_cnot: bool = True,
    init_all_left: bool = False,
    avoid_postselection_crossings: bool = True,
    z_only: bool = False,
    strict_hub: bool = False
):
    return StimToPyZXExtractor(
        open_inputs = open_inputs,
        postselect = postselect,
        init = init,
        prep_color = prep_color,
        serialize_cnot = serialize_cnot,
        init_all_left = init_all_left,
        avoid_postselection_crossings = avoid_postselection_crossings,
        z_only = z_only,
        cat_library = cat_library,
        t = t,
        spidercat_hub = spidercat_hub,
        min_weight = min_weight,
        strict_hub = strict_hub
    ).extract(circuit)


def mpp_to_cat_zx(mpp_circuit, library: CatLibrary, t: int | None = None, distance: int | None = None, min_weight: int = 4, **kwargs):
    if t is None:
        if distance is None:
            raise ValueError("give t = ... or distance = ...")
        t = distance // 2
    if isinstance(mpp_circuit, Path) or (isinstance(mpp_circuit, str) and "\n" not in mpp_circuit and len(mpp_circuit) < 4096
                                         and Path(mpp_circuit).is_file()):
        mpp_circuit = stim.Circuit(Path(mpp_circuit).read_text())
    return stim_to_pyzx(mpp_circuit, cat_library = library, t = t, spidercat_hub = True, min_weight = min_weight, **kwargs)


if __name__ == "__main__":
    lib = CatLibrary(sys.argv[1] if len(sys.argv) > 1 else None)

    print("check: spliced marked graph == a single n-legged Z-spider")
    for (n, t) in [(8, 2), (9, 2), (10, 3), (12, 3)]:
        got = lib.marked_graph(n, t)
        if got is None:
            print(f"  n = {n} t = {t}: no graph on disk"); continue
        G, M, t_ach, src = got

        g = zx.Graph()
        vmap = {v: g.add_vertex(VertexType.Z, qubit = 0, row = i) for i, v in enumerate(sorted(G.nodes()))}
        outs = []
        for (u, v) in sorted(tuple(sorted(e)) for e in G.edges()):
            prev, marks = vmap[u], M.get((u, v), 0)
            for _ in range(marks):
                b = g.add_vertex(VertexType.Z, qubit = 1, row = len(outs))
                g.add_edge((prev, b), EdgeType.SIMPLE)
                o = g.add_vertex(VertexType.BOUNDARY, qubit = 2, row = len(outs))
                g.add_edge((b, o), EdgeType.SIMPLE)
                outs.append(o)
                prev = b
            g.add_edge((prev, vmap[v]), EdgeType.SIMPLE)
        g.set_inputs(()); g.set_outputs(tuple(outs))

        ideal = zx.Graph()
        hub = ideal.add_vertex(VertexType.Z, qubit = 0, row = 0)
        io = []
        for i in range(len(outs)):
            o = ideal.add_vertex(VertexType.BOUNDARY, qubit = 1, row = i)
            ideal.add_edge((hub, o), EdgeType.SIMPLE)
            io.append(o)
        ideal.set_inputs(()); ideal.set_outputs(tuple(io))

        same = zx.compare_tensors(g, ideal, preserve_scalar = False)
        print(f"n = {n:>2} t = {t} -> marks = {len(outs)} V = {G.number_of_nodes()}"
              f"E = {G.number_of_edges()} ratio = {G.number_of_nodes()/len(outs):.3f} equal = {same}")
        assert same
    print("ok")
