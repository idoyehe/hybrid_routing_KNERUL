import networkx as nx
from consts import EdgeConsts
import matplotlib.pyplot as plt


def _star():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([
        (1, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (2, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (3, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (4, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (5, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
    ])

    return g


def _ring():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (2, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (5, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
    ])

    return g


def _mesh():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (1, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (2, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (0, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
    ])

    return g


def _fcn():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))

    for n1 in g.nodes:
        for n2 in range(n1 + 1, len(g.nodes)):
            g.add_edge(n1, n2, **{EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10})

    assert len(g.edges) == len(g.nodes) * (len(g.nodes) - 1) / 2
    return g


def _tree():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))

    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (1, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (2, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (2, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
    ])
    return g


def _line():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))

    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (2, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
        (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
    ])
    nx.draw(g)
    plt.show()
    return g


topologies = {
    "STAR": _star(),
    "RING": _ring(),
    "MESH": _mesh(),
    "FCN": _fcn(),
    "TREE": _tree(),
    "LINE": _line(),
}
