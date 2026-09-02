from collections import Counter

from Utils.print_OLA_graphs import plot

from .fragments import FRAGMENTS
from .occupancy import occupancy_profile
from .primitives import build_adjacency

__all__ = [
    "ramp_table",
    "ramp_comparison_table",
    "plot_ramp",
    "plot_ramps",
    "show_best"
]


def _frame(rows):
    try:
        import pandas as pd
        return pd.DataFrame([{k: v for k, v in r.items() if k not in ("schedule", "graph", "adj_w", "colour_map", "pos", "occupancy")}
                             for r in rows])
    except ImportError:
        return rows


def ramp_table(result):
    rows = [r for r in result["ramp"] if r["feasible"]]
    print(f"  {'q':>4}{'columns':>9}{'layers':>8}{'qubits':>8}{'volume':>8}"
          f"{'idle':>7}{'sec':>7}")
    for r in rows:
        print(f"  {r['q']:>4}{r['depth']:>9}{r['layers']:>8}{r['peak']:>8}"
              f"{r['volume']:>8}{r['idle']:>7}{r['seconds']:>7.1f}")
    return _frame(rows)


def ramp_comparison_table(results):
    qs = sorted({r["q"] for res in results.values() for r in res["ramp"]})
    print(f"  {'scheduler':<12}" + "".join(f"{q:>6}" for q in qs))
    for name, res in results.items():
        by_q = {r["q"]: r for r in res["ramp"] if r["feasible"]}
        print(f"  {name:<12}"
              + "".join(f"{(by_q[q]['depth'] if q in by_q else '-'):>6}"
                        for q in qs))
    print("  ('-' means no feasible schedule at that budget)")
    return _frame([r for res in results.values() for r in res["ramp"]
                   if r["feasible"]])


def plot_ramp(result):
    return plot_ramps({result.get("scheduler", "ramp"): result})


def plot_ramps(results):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    any_row = None
    for name, res in results.items():
        rows = [r for r in res["ramp"] if r["feasible"]]
        if not rows:
            continue
        any_row = res
        ax[0].plot([r["q"] for r in rows], [r["depth"] for r in rows], marker="o", ms=3, label=name)
        ax[1].plot([r["q"] for r in rows], [r["idle"] for r in rows], marker="o", ms=3, label=name)
    if any_row is None:
        print("nothing feasible to plot")
        return
    ax[0].axhline(any_row["parallel_limit"]["depth"], ls="--", color="crimson", lw=1, label="unlimited budget")
    ax[0].axvline(any_row["floor"], ls=":", color="grey", lw=1, label=f"floor q={any_row['floor']}")
    ax[0].set_xlabel("qubit budget q")
    ax[0].set_ylabel("circuit columns")
    ax[0].set_title("depth bought per qubit")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("qubit budget q")
    ax[1].set_ylabel("idle qubit-columns")
    ax[1].set_title("budget reserved but not doing work")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def show_best(result, label, allow_co_measure = False, polish_volume = None, unmerge_at_boundary = None, **plot_kwargs):
    best = result["best"]
    if best is None:
        print(f"{label}: no feasible schedule up to q_max")
        return None

    sched, G_best = best["schedule"], best["graph"]
    _, adj_w = build_adjacency(G_best)
    prof = occupancy_profile(G_best, sched, adj_w)

    plot(G_best, order = sched, cutwidths = best["peak"], depth = prof.depth,
         title_prefix = (f"{label} depth={prof.depth} | vol={polish_volume} | "
                       f"unmerge={unmerge_at_boundary} | co_meas={allow_co_measure}"), size = 200, **plot_kwargs)

    print(f"{label}: qubits = {prof.peak}, circuit columns = {prof.depth} volume = {prof.volume}")
    print(f"occupancy {prof.occ}")
    print(f"columns {prof.columns[0]} - {prof.columns[1]}")
    print(f"fragments {dict(Counter(FRAGMENTS[k][0] for k in prof.typings.values()))}")
    return prof
