def order_nodes(graph, order):
    timestep = {v: i for i, v in enumerate(order)}
    time_pos = {}
    for v in graph.nodes:
        x = timestep[v]          
        y = 0
        time_pos[v] = (x, y)
        graph.nodes[v]["t"] = x     
    return time_pos

def order_nodes_graphing(graph, order):
    timestep = {v: i for i, v in enumerate(order)}


    pos = {}
    for v in graph.nodes:
        pos[v] = (timestep[v], 0)
        graph.nodes[v]["t"] = timestep[v]

    return pos