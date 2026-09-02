from .layers import joint_groups_in_layer, leg_directions_ok
from .occupancy import occupancy_profile
from .primitives import no_one_sided
from .refine import move_earliest

__all__ = ["co_measure_cleanup"]


def _append_member(sched, group, G, adj_w, limit, max_fan, allow_co_measure):
    # last resort: isolate one member of group in a fresh trailing layer
    L = len(sched)
    for strict in (True, False):
        for mover in group:
            trial = [s[:] for s in sched]
            trial[L - 1].remove(mover)
            trial.append([mover])
            trial = [s for s in trial if s]
            if not leg_directions_ok(trial):
                continue
            if strict:
                prof = occupancy_profile(G, trial, adj_w, limit)
                if not (prof.feasible and no_one_sided(adj_w, trial, max_fan)):
                    continue
            return trial
    # forced: an invalid last-layer co-measure is worse and callers re-check
    trial = [s[:] for s in sched]
    trial[L - 1].remove(group[0])
    trial.append([group[0]])
    return [s for s in trial if s]


def _clear_last_layer_groups(sched, G, adj, adj_w, limit, max_fan, allow_co_measure):
    # on return the last layer contains no co-measured group
    for _ in range(100000):
        sched = [s for s in sched if s]
        if not sched:
            return sched
        L = len(sched)
        groups = joint_groups_in_layer(sched[L - 1], adj_w, G)
        if not groups:
            return sched
        group = groups[0]

        moved = None
        # 1. earliest earlier layer -> one-sided-clean first then relaxed
        for strict in (True, False):
            for mover in group:
                moved = move_earliest(G, sched, adj_w, adj, mover, L - 1, limit, max_fan=max_fan,
                                      allow_co_measure=allow_co_measure, require_no_one_sided=strict)
                if moved is not None:
                    break
            if moved is not None:
                break
        # 2. unavoidable: spend a layer
        if moved is None:
            moved = _append_member(sched, group, G, adj_w, limit, max_fan, allow_co_measure)
        sched = moved
    return [s for s in sched if s]


def co_measure_cleanup(G, schedule, adj_w, adj, limit, max_fan=2, allow_co_measure=True,
                       drop_non_reducing=True, allow_last_layer_measure=True):
    if not allow_co_measure:
        return [s for s in schedule if s]
    sched = [s[:] for s in schedule if s]

    if not allow_last_layer_measure:
        sched = _clear_last_layer_groups(sched, G, adj, adj_w, limit, max_fan,
                                         allow_co_measure)

    # drop co-measures that are not holding the depth -> a member that peels into an existing layer for free was never buying a layer
    # so separate it and hand the ancilla back
    # groups that cannot be separated without adding a layer are genuinely depth-reducing and are kept
    if drop_non_reducing:
        for _ in range(100000):
            progressed = False
            for t in range(len(sched)):
                for group in joint_groups_in_layer(sched[t], adj_w, G):
                    for mover in group:
                        res = move_earliest(
                            G, sched, adj_w, adj, mover, t, limit, max_fan=max_fan,
                            allow_co_measure=allow_co_measure,
                            allow_last_layer_measure=allow_last_layer_measure,
                            require_no_one_sided=True, pure_parallel=True)
                        if res is not None:
                            sched, progressed = res, True
                            break
                    if progressed:
                        break
                if progressed:
                    break
            if not progressed:
                break
        # peeling can shift another group into the last layer -> re-clear
        if not allow_last_layer_measure:
            sched = _clear_last_layer_groups(sched, G, adj, adj_w, limit, max_fan,
                                             allow_co_measure)

    return [s for s in sched if s]
