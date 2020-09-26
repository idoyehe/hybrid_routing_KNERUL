import networkx as nx
from common.consts import EdgeConsts
import matplotlib.pyplot as plt
import urllib.request
from Learning_to_Route.common.size_consts import SizeConsts

DEFAULT_CAPACITY = 10


def _star():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([
        (1, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (2, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (3, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (4, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (5, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
    ])
    g.graph["Name"] = "Star_6_Nodes"
    return g


def _ring():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (2, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (5, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
    ])
    g.graph["Name"] = "Ring_6_Nodes"
    return g


def _mesh():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))
    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (1, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (2, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (0, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
    ])
    g.graph["Name"] = "Mesh_6_Nodes"
    return g


def _fcn():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))

    for n1 in g.nodes:
        for n2 in range(n1 + 1, len(g.nodes)):
            g.add_edge(n1, n2, **{EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY})

    assert len(g.edges) == len(g.nodes) * (len(g.nodes) - 1) / 2
    g.graph["Name"] = "FCN_6_Nodes"
    return g


def _tree():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))

    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (1, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (2, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (2, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
    ])
    g.graph["Name"] = "Tree_6_Nodes"
    return g


def _line():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from(range(6))

    g.add_edges_from([
        (0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (2, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
        (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
    ])
    g.graph["Name"] = "Line_6_Nodes"
    return g


def _clique():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5])
    g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (0, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (0, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (0, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (1, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (1, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (1, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (2, 3, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (2, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (2, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (3, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY})])
    g.graph["Name"] = "Clique_3_Nodes"
    return g


def _triangle():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY}),
                      (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: DEFAULT_CAPACITY + 5})])
    g.graph["Name"] = "Triangle_3_Nodes"
    return g


def topology_zoo_loader(url: str, default_capacity: int = SizeConsts.ONE_Gb / SizeConsts.ONE_Mb):
    CAPACITY_LABEL_DEFAULT: str = "LinkSpeedRaw"
    if url.startswith("http"):
        gml = urllib.request.urlopen(str(url)).read().decode("utf-8")
    else:
        local_path = url
        from platform import system
        if system() == "Linux":
            local_path = "/home/idoye/PycharmProjects/Research_Implementing" + \
                         local_path.replace("\\", "/").split("/Research_Implementing")[1]

        gml_file = open(local_path, "r")
        gml = "".join(gml_file.readlines())
        gml_file.close()

    raw_g = nx.parse_gml(gml, label="id")
    if "link_weights_graph" in raw_g.graph:
        parsed_g = nx.DiGraph()
        parsed_g.add_nodes_from(raw_g.nodes)
        parsed_g.add_edges_from(raw_g.edges.data())
    else:
        parsed_g = nx.Graph()
        parsed_g.add_nodes_from(raw_g.nodes)
        for raw_edge in raw_g.edges:
            edge = (raw_edge[0], raw_edge[1])
            if edge not in parsed_g.edges:
                parsed_g.add_edge(u_of_edge=edge[0], v_of_edge=edge[1])
                parsed_g.edges[edge][EdgeConsts.CAPACITY_STR] = 0
            if CAPACITY_LABEL_DEFAULT in raw_g.edges[raw_edge]:
                raw_capacity = int(raw_g.edges[raw_edge][CAPACITY_LABEL_DEFAULT]) / SizeConsts.ONE_Mb
                parsed_g.edges[edge][EdgeConsts.CAPACITY_STR] += raw_capacity
            else:
                parsed_g.edges[edge][EdgeConsts.CAPACITY_STR] += default_capacity

    parsed_g.graph["Name"] = raw_g.graph["Network"]
    return parsed_g


BASIC_TOPOLOGIES = {
    "STAR": _star(),
    "RING": _ring(),
    "MESH": _mesh(),
    "FCN": _fcn(),
    "TREE": _tree(),
    "LINE": _line(),
    "TRIANGLE": _triangle(),
    "CLIQUE": _clique()
}
