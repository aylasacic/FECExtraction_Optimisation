# annealing temperature schedule
T_INIT = 10
T_MIN = 1e-6
ALPHA = 0.995

# qubit budget the ramp is allowed to spend
QUBIT_LIMIT = 1000

# unfuse one-sided high-fan spiders
UNMERGE_AT_BOUNDARY = False

# put adjacent same-colour spiders in one layer as a joint measurement.
# progressive_depth_search still refuses this: a same-layer edge has no time
# direction, so neither endpoint gets a fragment typing.
ALLOW_CO_MEASURE = False

# second pass: minimise volume at fixed depth
POLISH_VOLUME = True

# SA seeds; e.g. (23, 24, 25, 26) to take the best of four
SEEDS = (23,)

SA_KWARGS = dict(
    T_init=T_INIT,
    T_min=T_MIN,
    alpha=ALPHA,
    steps_per_temp=20,
    prob_adj=0.6,
)
