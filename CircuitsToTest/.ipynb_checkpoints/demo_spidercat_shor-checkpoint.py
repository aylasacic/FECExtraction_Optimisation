"""
demo_spidercat_shor.py -- end-to-end walkthrough.

    python demo_spidercat_shor.py --spidercat spidercat-main/spidercat \
                                  --circuits circuits

Everything here is also usable interactively; each section is independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import stim

from spidercat_shor import (CatLibrary, MppCatRewriter, parse_pauli_products, shor_gadget, verify_fault_tolerance, code_params_from_name, _mpp_targets)

EXAMPLE = """\
MPP X0*Z10*Z11*X12*X13*Z15*X16*Z17*X18*Y19*Y20*X21*X23*X24
MPP Z0*Z10*Y11*Z12*Z13*Y17*Y19*Z21*X22*Z23*Z24
MPP Z7*Z12*X13*X17*Z18*Y21*X23
MPP X10*Y15*Z16*Z18*Z19*X20*X22
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spidercat", default=None,
                    help="folder holding circuits/ and circuits_data/")
    ap.add_argument("--circuits", default=None,
                    help="folder of your *_mpp.stim syndrome files")
    ap.add_argument("--out", default="circuits_cat")
    ap.add_argument("--zx", action="store_true", help="also build PyZX diagrams")
    args = ap.parse_args(argv)

    lib = CatLibrary(args.spidercat)
    print(f"SpiderCat root: {lib.root}")

    # ------------------------------------------------------------------ #
    print("\n[1] one gadget, inspected")
    terms = parse_pauli_products(stim.Circuit(EXAMPLE.splitlines()[0]).flattened()[0].targets_copy())[0]
    cat = lib.cat_circuit(len(terms), t=3)
    print(f"    product  : {'*'.join(str(t) for t in terms)}")
    print(f"    weight   : {cat.n}   t: {cat.t} (requested 3)   flags: {cat.num_flags}")
    print(f"    legs     : {cat.legs}")
    print(f"    source   : {cat.source}")
    print(f"    is |CAT>?: {lib.verify(cat)}")
    gadget, syn = shor_gadget(cat, terms, ancilla_base=25)
    print("    gadget (first lines):")
    for line in str(gadget).splitlines()[:6]:
        print("      " + line)
    print(f"      ...   syndrome = XOR of rec{syn}")

    # ------------------------------------------------------------------ #
    print("\n[2] the cat really has the claimed distance")
    r = verify_fault_tolerance(cat, max_weight=2)
    print(f"    exhaustive over {r['faults_tested']} faults at {r['num_fault_locations']} "
          f"locations: ok={r['ok']}, worst propagated weight={r['worst_output_weight']}")

    # ------------------------------------------------------------------ #
    print("\n[3] whole-file rewrite")
    rw = MppCatRewriter(lib, distance=7, ancilla_mode="reuse", syndrome="observable")
    res = rw.rewrite(EXAMPLE)
    print(res.summary())

    # ------------------------------------------------------------------ #
    print("\n[4] the rewritten gadgets agree with the original MPPs")
    check = stim.Tableau.random(25).to_circuit("elimination")
    for terms in [p for op in stim.Circuit(EXAMPLE).flattened() if op.name == "MPP"
                  for p in parse_pauli_products(op.targets_copy())]:
        c = lib.cat_circuit(len(terms), 3)
        g, off = shor_gadget(c, terms, ancilla_base=25)
        check += g
        check.append("MPP", _mpp_targets(terms))
        check.append("DETECTOR",
                     [stim.target_rec(o - 1) for o in off] + [stim.target_rec(-1)])
    check.detector_error_model(decompose_errors=False)
    fired = check.compile_detector_sampler().sample(500).any()
    print(f"    500 shots on a random stabiliser state, any mismatch: {fired}")

    # ------------------------------------------------------------------ #
    if args.circuits:
        print(f"\n[5] batch: {args.circuits} -> {args.out}")
        out = MppCatRewriter(lib, t=1).rewrite_folder(args.circuits, args.out)
        for name, r in out.items():
            d = code_params_from_name(name).get("d")
            print(f"    {name}: d={d} t={(d or 2)//2} "
                  f"{len(r.gadgets)} checks, {r.num_qubits} qubits")

    # ------------------------------------------------------------------ #
    if args.zx:
        print("\n[6] PyZX")
        import pyzx as zx
        from cat_zx import mpp_to_cat_zx, stim_to_pyzx

        g_b, info_b = mpp_to_cat_zx(EXAMPLE, lib, distance=7)
        print(f"    route B (graph spliced into the hub): {g_b.num_vertices()} vertices, "
              f"max arity {max(g_b.vertex_degree(v) for v in g_b.vertices())}")
        for h in info_b["cat_hubs"]:
            print(f"      weight {h['weight']:>2}  t={h['t']}  V={h['num_vertices']} "
                  f"ratio={h['vertex_ratio']:.3f}  {h['source']}")
        g_a, _ = stim_to_pyzx(res.circuit, open_inputs=False, postselect=True,
                                  serialize_cnot=True, avoid_postselection_crossings=True)
        print(f"    route A (via the rewritten stim circuit): {g_a.num_vertices()} vertices, "
              f"max arity {max(g_a.vertex_degree(v) for v in g_a.vertices())}")
        print("    (use zx.draw(g_b) in a notebook to look at it)")

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
