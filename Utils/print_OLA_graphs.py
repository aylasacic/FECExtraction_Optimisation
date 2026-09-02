import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Utils.ordering import order_nodes_graphing, order_nodes
from collections import Counter, defaultdict

NODE_COLOR = "#000000"
IO_COLOR = "#ffffff"
EDGE = "#000000"
NODE_BORDER = "#000000"
LABEL_COLOR = "#000000"
ZX_FILL = {"Z": "#5bc85b", "X": "#f26d6d", "BOUNDARY": "#ffffff"}
DEFAULT_FILL = "#000000" 

def node_category(graph, n):
    if n in {"I", "O"}:   
        return "BOUNDARY"
    vt = graph.nodes[n].get("vtype")
    if vt in ZX_FILL:
        return vt
    key = graph.nodes[n].get("key", "") 
    if isinstance(key, str) and key[:3] in ("BI_", "BO_"):
        return "BOUNDARY"
    return None

def is_pos_dict(d):
    return (isinstance(d, dict) and bool(d) and all(isinstance(v, (tuple, list)) and len(v) == 2 for v in d.values()))

def max_column_size(ordering):
    # schedule = list of timesteps
    if isinstance(ordering, list):
        return max((len(timestep) for timestep in ordering), default=1)

    # position dict = {"node": (x, y)}
    if is_pos_dict(ordering):
        columns = Counter(round(x, 6) for x, _ in ordering.values())
        return max(columns.values(), default = 4)

    return 1

def ordering_column_count(ordering):
    if is_pos_dict(ordering):
        return len({round(x, 6) for x, _ in ordering.values()})

    if isinstance(ordering, list):
        return max(1, len(ordering))

    return 1
    
# def schedule_to_pos(graph, schedule, x_gap = 1.0, y_gap = 1.0):
#     pos = {"I": (-x_gap, 0)}

#     for t, timestep in enumerate(schedule):
#         n = len(timestep)

#         for i, node in enumerate(timestep):
#             # Center every depth column around y=0
#             y = ((n - 1) / 2 - i) * y_gap
#             pos[node] = (t * x_gap, y)

#     pos["O"] = (len(schedule) * x_gap, 0)

#     return pos


def schedule_to_pos(graph, schedule, depth = None, x_gap = 1.0, y_gap = 1.0, x_gap_factor = 4, y_gap_factor = 10, align = True):
    depth = max(depth or len(schedule), 1)

    x_gap = x_gap_factor * graph.number_of_nodes() / depth
    y_gap = y_gap_factor * graph.number_of_nodes() / depth
    if not align:
        pos = {"I": (-x_gap, 0)}
        for t, timestep in enumerate(schedule):
            n = len(timestep)
            for i, node in enumerate(timestep):
                pos[node] = (t * x_gap, ((n - 1) / 2 - i) * y_gap)
        pos["O"] = (len(schedule) * x_gap, 0)
        return pos

    # timestep of every scheduled node
    layer_of = {}
    for t, timestep in enumerate(schedule):
        for node in timestep:
            layer_of[node] = t

    # boundary anchored on the center track (y = 0)
    track_of = {"I": 0}          

    def find_free_track(pref, used):
        # nearest free integer track to `pref`, searching outward so it stays balanced
        base = int(round(pref))
        d = 0
        while True:
            for cand in ((base,) if d == 0 else (base - d, base + d)):
                if cand not in used:
                    return cand
            d += 1

    for t, timestep in enumerate(schedule):
        used = set()
        # nodes that couldnt align this pass
        pending = []                          
        # pass 1: greedily inherit a previous-column neighbours track (-> straight edge)
        for node in timestep:
            nbr_tracks = []
            for nbr in graph.neighbors(node):
                if nbr in track_of and layer_of.get(nbr, -2) == t - 1:
                    nbr_tracks.append(track_of[nbr])
                elif nbr == "I" and t == 0:
                    nbr_tracks.append(0)
            chosen = next((tr for tr in nbr_tracks if tr not in used), None)
            if chosen is not None:
                used.add(chosen)
                track_of[node] = chosen
            else:
                pref = sum(nbr_tracks) / len(nbr_tracks) if nbr_tracks else 0.0
                pending.append((node, pref))
        # pass 2: everyone else gets their own nearest free track -> no overlap
        for node, pref in pending:
            tr = find_free_track(pref, used)
            used.add(tr)
            track_of[node] = tr

    pos = {"I": (-x_gap, 0)}
    for node, t in layer_of.items():
        pos[node] = (t * x_gap, track_of[node] * y_gap)
    pos["O"] = (len(schedule) * x_gap, 0)
    return pos

def adaptive_rad_DO(u, v, pos, min_rad = 0.08, max_rad = 0.45, curviness = None):
    (x1, y1), (x2, y2) = pos[u], pos[v]
    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist == 0:
        return max_rad
    rad = max(min_rad, min(max_rad, max_rad / (1 + dist)))
    if abs(dx) >= abs(dy):
        sign = 1 if dx > 0 else -1
    else:
        sign = 1 if dy > 0 else -1
    return sign * rad

def adaptive_rad_LO(u, v, pos, curviness = 1.0, min_rad = 0.15, max_rad = 0.7):
    (x1, y1), (x2, y2) = pos[u], pos[v]
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        return max_rad if dy >= 0 else -max_rad
    span = abs(dx)
    rad = curviness * min(max_rad, min_rad + 0.07 * (span - 1))
    return rad if dx > 0 else -rad


def draw_adaptive_edges(graph, pos, ax, style = "LO", highlight_backward = True, node_size = 420):
    rad_fn = adaptive_rad_LO if style == "LO" else adaptive_rad_DO
    groups = defaultdict(list) # just to create a list of values because the graph pasted sometime turns into a set??? IDK WHY

    if graph.is_multigraph():
        for u, v, k in graph.edges(keys=True):
            if graph.is_directed():
                key = (u, v)
            else:
                key = frozenset((u, v))
            groups[key].append((u, v, k))
    else:
        for u, v in graph.edges():
            if graph.is_directed():
                key = (u, v)
            else:
                key = frozenset((u, v))
            groups[key].append((u, v, None))

    # for edges in groups.values():
    #     n = len(edges)
    #     offsets = [0.0] if n == 1 else np.linspace(-0.25, 0.25, n)

    #     for (u, v, k), off in zip(edges, offsets):
    #         base_rad = rad_fn(u, v, pos, curviness=0.75)
    #         rad = base_rad + off

    #         nx.draw_networkx_edges(graph, pos, edgelist = [(u, v)], connectionstyle = f"arc3,rad={rad}", edge_color = EDGE, 
    #                                width = 1.25, alpha = 0.75, arrows = True, arrowstyle = "-", node_size = node_size, 
    #                                min_source_margin = 0, min_target_margin = 0, ax = ax,)
    for edges in groups.values():
        n = len(edges)
        offsets = [0.0] if n == 1 else np.linspace(-0.25, 0.25, n)

        for (u, v, k), off in zip(edges, offsets):
            same_track = abs(pos[u][1] - pos[v][1]) < 1e-9
            if same_track:
                rad = 0.06 + off * 0.3
            else:
                base_rad = rad_fn(u, v, pos, curviness = 0.75)
                rad = base_rad + off

            nx.draw_networkx_edges(graph, pos, edgelist = [(u, v)], connectionstyle = f"arc3,rad={rad}", edge_color = EDGE,
                                   width = 1.25, alpha = 0.75, arrows = True, arrowstyle = "-", node_size = node_size,
                                   min_source_margin = 0, min_target_margin = 0, ax = ax)

def _infer_style(pos):
    cols = Counter(round(x, 6) for x, _ in pos.values())
    return "DO" if any(c > 1 for c in cols.values()) else "LO"

def plot(graph, order, cutwidths = None, depth = None, styles = None, title_prefix = "", size = 420, 
         draw_labels = False, align = True, colour_in = False):
    if isinstance(order, dict) and not is_pos_dict(order):
        items = list(order.items())
        cut_for = (lambda l: cutwidths.get(l)) if isinstance(cutwidths, dict) else (lambda l: cutwidths)
        style_for = (lambda l: styles.get(l)) if isinstance(styles, dict) else (lambda l: styles)
    else:
        items = [(title_prefix, order)]
        cut_for = lambda l: cutwidths
        style_for = lambda l: styles

    regular_nodes = [node for node in graph.nodes if node not in {"I", "O"}]
    io_nodes = [node for node in graph.nodes if node in {"I", "O"}]

    node_colors = [IO_COLOR if n in {"I", "O"} else NODE_COLOR for n in graph.nodes]
    rc = {"figure.facecolor": "white", "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": "#22333b"}
    with plt.rc_context(rc):
        n = len(items)
        panel_heights = [min(10, max(4.5, 1.5 + 0.25 * max_column_size(ordering))) for _, ordering in items]
        max_columns = max(ordering_column_count(ordering) for _, ordering in items)
        max_rows = max(max_column_size(ordering) for _, ordering in items)
        figure_width = min(24, max(14, 0.9 * max_columns, 0.45 * max_rows))
        fig, axes = plt.subplots(n, 1, figsize = (figure_width, sum(panel_heights)), gridspec_kw = {"height_ratios": panel_heights})
        axes = np.atleast_1d(axes)
        for ax, (label, ordering) in zip(axes, items):
            # print(ordering if isinstance(ordering, dict) else schedule_to_pos(graph, ordering, depth = len(order), align = align))
            pos = ordering if isinstance(ordering, dict) else schedule_to_pos(graph, ordering, depth = depth, align = align)
            style = style_for(label) or _infer_style(pos)
            cw = cut_for(label)
            title = label or title_prefix
            if cw is not None:
                title = f"{title} | max qubits = {cw}"
            ax.set_title(title, pad = 10)
            ax.axis("off")
            ax.margins(x = 0.06, y = 0.12)
            draw_adaptive_edges(graph, pos, ax, style = style, node_size = size)
            if colour_in:
                by_type = defaultdict(list)
                for nd in graph.nodes:
                    by_type[node_category(graph, nd)].append(nd)
    
                for cat, fill in ZX_FILL.items():
                    nodes = by_type.get(cat, [])
                    if not nodes:
                        continue
                    is_boundary = cat == "BOUNDARY"
                    nx.draw_networkx_nodes(graph, pos, nodelist = nodes, node_size = size * (0.55 if is_boundary else 1.0),
                                           node_color = fill, edgecolors = NODE_BORDER if is_boundary else "#222222",
                                           linewidths = 1.0 if is_boundary else 0.6, ax = ax)
    
                unknown = by_type.get(None, [])
                if unknown:
                    nx.draw_networkx_nodes(graph, pos, nodelist = unknown, node_size = size,
                                           node_color = DEFAULT_FILL, edgecolors = "none",
                                           linewidths = 0, ax = ax)
            else:
            # nx.draw_networkx_nodes(graph, pos, node_size = size, node_color = node_colors, edgecolors = NODE_BORDER, ax = ax)
                nx.draw_networkx_nodes(graph, pos, nodelist = regular_nodes, node_size = size, node_color = NODE_COLOR, 
                                       edgecolors = "none", linewidths = 0, ax = ax)
                nx.draw_networkx_nodes(graph, pos, nodelist = io_nodes, node_size = size, node_color = IO_COLOR,
                                       edgecolors = NODE_BORDER, linewidths = 1.0, ax = ax)
            if draw_labels:
                nx.draw_networkx_labels(graph, pos, font_size = 7, font_weight = "bold", font_color = LABEL_COLOR, ax = ax)
        fig.tight_layout()
        plt.show()