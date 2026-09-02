from . import config
from .fragments import (
    FOOTPRINT,
    FRAGMENTS,
    NotExtractable,
    Occupancy,
    REACH_BACK,
    REACH_FWD,
    check_fragment_table,
    spider_typing,
)
from .primitives import (
    build_adjacency,
    can_place,
    first_high_fan,
    lr_degrees,
    no_one_sided,
    same_colour,
    spider_colour,
    spider_io,
    timesteps,
    total_energy,
    vtype_name,
)
from .occupancy import (
    check_no_same_layer_edges,
    check_no_self_loops,
    check_schedule_covers_graph,
    explain_column,
    occupancy_profile,
)
from .order_cache import (
    Proposal,
    commit_order,
    make_occ_cache,
    occ_from_delta,
    order_extent,
    propose_order_move,
    propose_order_swap,
    stamp_order,
    validate_occ_cache,
)
from .layers import (
    components,
    greedy_layers,
    joint_groups_in_layer,
    leg_directions_ok,
    vert_info,
)
from .refine import move_earliest, refine, try_move
from .co_measure import co_measure_cleanup
from .unfusion import optimize_depth_with_unfusion, relink_adjacency, unfuse
from .pipeline import Sched, as_sched, schedule_min_depth
from .schedulers import (
    sched_anneal,
    sched_asap,
    sched_cap_sweep,
    sched_capped,
    sched_serial,
)

# these reach outside the package
from .annealing import simulated_annealing_feasibility          # Algorithms.sa
from .ramp import compare_ramps, progressive_depth_search        # Utils.print_OLA_graphs
from .reporting import (                                         # Utils.print_OLA_graphs
    plot_ramp,
    plot_ramps,
    ramp_comparison_table,
    ramp_table,
    show_best,
)
from .extraction import (                                        # CircuitExtraction
    EdgeRef,
    ScheduleExtractionResult,
    extract_schedule,
)
from .stim_backend import StimExtractor, schedule_to_stim        # stim

__all__ = [
    "config",
    # fragments
    "FRAGMENTS", "FOOTPRINT", "REACH_BACK", "REACH_FWD", "Occupancy",
    "NotExtractable", "spider_typing", "check_fragment_table",
    # primitives
    "build_adjacency", "timesteps", "spider_io", "first_high_fan",
    "no_one_sided", "vtype_name", "spider_colour", "same_colour", "can_place",
    "lr_degrees", "total_energy",
    # occupancy
    "occupancy_profile", "explain_column", "check_no_self_loops",
    "check_schedule_covers_graph", "check_no_same_layer_edges",
    # order space
    "Proposal", "stamp_order", "order_extent", "occ_from_delta",
    "make_occ_cache", "propose_order_swap", "propose_order_move",
    "commit_order", "validate_occ_cache", "simulated_annealing_feasibility",
    # layering and refinement
    "vert_info", "leg_directions_ok", "greedy_layers", "components",
    "joint_groups_in_layer", "try_move", "refine", "move_earliest",
    "co_measure_cleanup", "relink_adjacency", "unfuse",
    "optimize_depth_with_unfusion", "Sched", "as_sched", "schedule_min_depth",
    # schedulers and the ramp
    "sched_serial", "sched_asap", "sched_capped", "sched_cap_sweep",
    "sched_anneal", "progressive_depth_search", "compare_ramps",
    # reporting
    "ramp_table", "ramp_comparison_table", "plot_ramp", "plot_ramps",
    "show_best",
    # extraction
    "EdgeRef", "ScheduleExtractionResult", "extract_schedule",
    "StimExtractor", "schedule_to_stim",
]
