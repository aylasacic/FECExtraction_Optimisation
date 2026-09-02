"""
FRAGMENT PROFILES

a spider with n_in inputs and n_out outputs runs max(n_in, n_out) lines
min(n_in, n_out) of them pass straight through and the rest are born (a split or a CAT preparation)
or die (a merge or a CAT effect)
reading the columns of each extraction circuit gives when:
 (0,3) 3-CAT       |+>,|0> prepared at t-2, third |0> at t-1, CNOTs at t-1 and t  -> 3 births at -2, -2, -1
 (0,2) 2-CAT       both prepared at t-1, CNOT at t, Clifford at t+1 -> 2 births at -1, -1
 (1,2) split       Clifford and fresh |0> at t-1, CNOT at t -> 1 birth at -1
 (1,1) gate        Clifford at t -> nothing born or dying
 (2,1) merge       CNOT at t, <0| and Clifford at t+1 -> 1 death at +1
 (2,0) 2-CAT adj   Clifford at t-1, CNOT at t, both measured at t+1 -> 2 deaths at +1, +1
 (3,0) 3-CAT adj   CNOT and Clifford at t, CNOT and <0| at t+1, last two measured at t+2 -> 3 deaths at +1, +2, +2
"""

from collections import namedtuple

__all__ = [
    "FRAGMENTS",
    "FOOTPRINT",
    "REACH_BACK",
    "REACH_FWD",
    "Occupancy",
    "NotExtractable",
    "spider_typing",
    "check_fragment_table"
]


FRAGMENTS = {
    #  (n_in, n_out): (name,             births,        deaths)
    (0, 3):           ("3-CAT",          (-2, -2, -1),  ()),
    (3, 0):           ("3-CAT adjoint",  (),            (+1, +2, +2)),
    (0, 2):           ("2-CAT",          (-1, -1),      ()),
    (2, 0):           ("2-CAT adjoint",  (),            (+1, +1)),
    (1, 2):           ("split",          (-1,),         ()),
    (2, 1):           ("merge",          (),            (+1,)),
    (1, 1):           ("gate",           (),            ()),
    # arity-one ancillas
    (0, 1):           ("ancilla prep",   (0,),          ()),
    (1, 0):           ("ancilla meas",   (),            (0,))
}

FOOTPRINT = {
                (0, 3): {-2: 2, -1: 3, 0: 3},
                (3, 0): {0: 3, +1: 3, +2: 2},
                (0, 2): {-1: 2, 0: 2, +1: 2},
                (2, 0): {-1: 2, 0: 2, +1: 2},
                (1, 2): {-2: 1, -1: 2},
                (2, 1): {+1: 2, +2: 1},
                (1, 1): {0: 1},
                (0, 1): {0: 1},
                (1, 0): {0: 1}
}

REACH_BACK = 2      # furthest a fragment reaches before its own layer
REACH_FWD = 2       # furthest it reaches after

Occupancy = namedtuple("Occupancy", "occ peak volume depth columns feasible typings")


class NotExtractable(ValueError):
    """ the schedule cannot be read off as a fault-equivalent circuit [TBD]"""


def spider_typing(io, node):
    #  (n_in, n_out) for a scheduled spider, checked against the seven+two
    key = io[node]
    if key not in FRAGMENTS:
        if sum(key) > 3:
            raise NotExtractable(
                f"spider {node!r} has arity {sum(key)} as {key}; the diagram is not 3-bounded, no extraction fragment")
        raise NotExtractable(f"spider {node} types as {key}, which is not one of the fragments")
    return key


def check_fragment_table():
    for key, (name, births, deaths) in FRAGMENTS.items():
        n_in, n_out = key
        through = min(n_in, n_out)
        want = FOOTPRINT[key]
        # only over the fragment's own span: outside it the lines are still alive, they just belong to whatever consumes them next
        got = {d: (through + sum(1 for b in births if b <= d) + sum(1 for x in deaths if x >= d)) for d in want}
        if got != want:
            raise AssertionError(
                f"{name} {key}: derived footprint {got} != stated {want}")
    return True
