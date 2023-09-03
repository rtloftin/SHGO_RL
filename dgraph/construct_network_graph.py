import networkx as nx


from .critical_points import Minimum, TransitionState
from .math_utils import is_saddle


def construct_network_graph(mins, func):
    edges = []
    vertices = []
    energy_cache = {}
    for i, v in enumerate(mins):
        min1 = Minimum(energy=v.f, coords=v.x, _id=v.index)
        vertices.append(min1)
        energy_cache[v.x] = v.f

        for j, v2 in enumerate(mins):
            if j <= i:
                continue
            coords = 0.5 * (v2.x_a - v.x_a) + v.x_a
            ts_energy = func(coords)
            if is_saddle(func, coords):
                min2 = Minimum(energy=v2.f, coords=v2.x, _id=v2.index)
                ts = TransitionState(energy=ts_energy, coords=coords, min1=min1, min2=min2)
                edges.append((min1, min2, {'ts': ts}))

    if len(edges) == 0:
        return None
    g = nx.Graph()
    g.add_nodes_from(vertices)
    g.add_edges_from(edges)
    return g