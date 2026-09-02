import argparse
import re
import sys
import traceback
from pathlib import Path
import stim


# PASTER FROM THE IPYNB

FNAME_RE = re.compile(r"q[_-]?(\d+)[_-]+n[_-]?(\d+)[_-]+k[_-]?(\d+)[_-]+d[_-]?(\d+)", re.I)
# OUT = Path("circuits")
# OUT.mkdir(parents=True, exist_ok=True)
PAULI = {(0, 0): "_", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}


def parse_params(stem):
    m = FNAME_RE.search(stem)
    return tuple(int(g) for g in m.groups()) if m else None

def parse_matrix(text):
    gens = []
    n = None
    for raw in text.splitlines():
        line = raw.strip().replace("[", "").replace("]", "")
        if "|" not in line:
            continue  # skip blanks / comments / headers
        xpart, zpart = line.split("|")
        x = [int(b) for b in xpart.split()]
        z = [int(b) for b in zpart.split()]
        if n is None:
            n = len(x)
        if len(x) != n or len(z) != n:
            raise ValueError(f"got {len(x)} X bits, {len(z)} Z bits")
        pauli = "".join(PAULI[(x[j], z[j])] for j in range(n))
        gens.append(stim.PauliString("+" + pauli))
    if not gens:
        raise ValueError("no '[X|Z]' rows found in file")
    return gens, n

def mpp_targets(ps):
    tgt = []
    # p in {0:I, 1:X, 2:Y, 3:Z}
    for q, p in enumerate(ps):  
        if p == 0:
            continue
        f = {1: stim.target_x, 2: stim.target_y, 3: stim.target_z}[p]
        if tgt:
            tgt.append(stim.target_combiner())
        # a negative-sign product flips its outcome: invert exactly one target
        tgt.append(f(q, invert=(ps.sign == -1 and not tgt)))
    return tgt


def build_mpp_directly(gens):
    c = stim.Circuit()
    for ps in gens:
        c.append("MPP", mpp_targets(ps))
    return c

def mpp_lines(gens):
    lines = []
    for ps in gens:
        c = stim.Circuit()
        c.append("MPP", mpp_targets(ps))
        lines.append(str(c).strip())
    return "\n".join(lines) + "\n"

def build_mpp_from_tableau(gens):
    tab = stim.Tableau.from_stabilizers(gens, allow_redundant = False, allow_underconstrained = True)
    c = stim.Circuit()
    for i in range(len(gens)):
        c.append("MPP", mpp_targets(tab.z_output(i)))
    return c, tab

def build_ancilla(gens, n):
    c = stim.Circuit()
    anc = {i: n + i for i in range(len(gens))}
    c.append("R", list(anc.values()))
    c.append("H", list(anc.values()))
    for i, s in enumerate(gens):
        a = anc[i]
        for q, p in enumerate(s):
            if p == 1:
                c.append("CX", [a, q])
            elif p == 2:
                c.append("CY", [a, q])
            elif p == 3:
                c.append("CZ", [a, q])
    c.append("H", list(anc.values()))
    c.append("M", list(anc.values()))
    return c

def process_file(path, outroot):
    stem = path.stem
    params = parse_params(stem)

    if params is None:
        return "skip", "filename does not match q_n_k_d pattern"
    q, n_name, k_name, d_name = params

    if q != 4:
        return "skip", f"q={q}: stim only models qubits (q=2)"

    gens, n = parse_matrix(path.read_text())
    m = len(gens)

    if n != n_name:
        print(f"! column count {n} != n={n_name} from filename (using matrix)")

    if not all(gens[i].commutes(gens[j]) for i in range(m) for j in range(i + 1, m)):
        return "fail", "generators do not pairwise commute (not a valid code)"

    try:
        mpp_tab, tab = build_mpp_from_tableau(gens)
    except ValueError as e:
        return "fail", f"stim rejected the generators: {e}"

    k = n - m
    if k != k_name:
        print(f"! derived k={k} (n-generators) != k={k_name} from filename")

    mpp_direct = build_mpp_directly(gens)
    ancilla = build_ancilla(gens, n)
    encoder = tab.to_circuit(method="graph_state")

    same = mpp_direct == mpp_tab

    outdir = outroot / stem
    outdir.mkdir(parents=True, exist_ok=True)
    # (outdir / "syndrome_mpp.stim").write_text(str(mpp_direct))
    (outdir / "syndrome_mpp.stim").write_text(mpp_lines(gens))
    (outdir / "syndrome_mpp_from_tableau.stim").write_text(str(mpp_tab))
    (outdir / "syndrome_ancilla.stim").write_text(str(ancilla))
    (outdir / "encoder.stim").write_text(str(encoder))
    (outdir / "stabilizers.txt").write_text("".join(f"g{i:3d}: {s}\n" for i, s in enumerate(gens, 1)))

    note = "" if same else "  (direct != tableau MPP; reordered/re-signed)"
    return "ok", f"[[{n},{k},{d_name}]] {m} generators{note}"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest = "indir", default = "stabmatrices/codetables_cache", help = "input folder of stabilizer-matrix .txt files")
    ap.add_argument("--out", dest = "outdir", default = "circuits", help = "output folder for generated circuits")
    ap.add_argument("--glob", default = "*", help = "filename glob to match (files have no extension by default)")
    args = ap.parse_args()
 
    indir = Path(args.indir)
    outroot = Path(args.outdir)
    if not indir.is_dir():
        sys.exit(f"no input folder: {indir}")
    outroot.mkdir(parents=True, exist_ok=True)
 
    files = sorted(p for p in indir.glob(args.glob) if p.is_file())
    if not files:
        sys.exit(f"no files matching {args.glob!r} in {indir}")
 
    tally = {"ok": 0, "skip": 0, "fail": 0}
    for path in files:
        print(path.name)
        try:
            status, msg = process_file(path, outroot)
        except Exception as e:
            status, msg = "fail", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        tally[status] += 1
        print(f"    {status.upper():4s} {msg}")
 
    print(f"\ndone: {tally['ok']} ok, {tally['skip']} skipped, "
          f"{tally['fail']} failed  ->  {outroot}/")
 
main()