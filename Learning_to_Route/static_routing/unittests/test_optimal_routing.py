import pytest
import numpy as np
from common.topologies import BASIC_TOPOLOGIES
from optimal_load_balancing import optimal_load_balancing_LP_solver
from common.network_class import NetworkClass


def test_star_topology():
    star_topo = NetworkClass(BASIC_TOPOLOGIES["STAR"])
    tm = np.zeros(shape=(star_topo.get_num_nodes, star_topo.get_num_nodes))
    for dst in star_topo.nodes():
        tm[0, dst] = 10
    tm[0, 0] = 0
    ratio, routing = optimal_load_balancing_LP_solver(star_topo, tm)
    assert ratio == 1
    for edge, edge_routing in routing.items():
        if edge[0] == 0:
            assert np.sum(edge_routing) == 1
            assert edge_routing[edge] == 1
        else:
            assert edge[1] == 0
            assert np.sum(edge_routing) == 0


def test_ring_topology():
    ring_topo = NetworkClass(BASIC_TOPOLOGIES["RING"])
    tm = np.zeros(shape=(ring_topo.get_num_nodes, ring_topo.get_num_nodes))
    for node in ring_topo.nodes():
        tm[node, (node + 1) % ring_topo.get_num_nodes] = 10
    ratio, routing = optimal_load_balancing_LP_solver(ring_topo, tm)
    assert ratio == 0.8333333333333339
    for edge, edge_routing in routing.items():
        assert np.sum(edge_routing) <= 0.8333333333333339


def test_fcn_topology():
    fcn_topo = NetworkClass(BASIC_TOPOLOGIES["FCN"])
    tm = np.zeros(shape=(fcn_topo.get_num_nodes, fcn_topo.get_num_nodes))
    for src, dst in fcn_topo.get_all_pairs():
        tm[src, dst] = 10

    ratio, routing = optimal_load_balancing_LP_solver(fcn_topo, tm)
    assert ratio == 1
    for edge, edge_routing in routing.items():
        assert edge_routing[edge] == 1
        assert np.sum(edge_routing) == 1


def test_tree_topology():
    tree_topo = NetworkClass(BASIC_TOPOLOGIES["TREE"])
    tm = np.zeros(shape=(tree_topo.get_num_nodes, tree_topo.get_num_nodes))
    for dst in tree_topo.nodes():
        tm[0, dst] = 10
    tm[0, 0] = 0

    ratio, routing = optimal_load_balancing_LP_solver(tree_topo, tm)
    assert ratio == 3
    assert np.sum(routing[(0, 1)]) == 2
    assert np.sum(routing[(0, 2)]) == 3


def test_line_topology():
    line_topo = NetworkClass(BASIC_TOPOLOGIES["LINE"])
    tm = np.zeros(shape=(line_topo.get_num_nodes, line_topo.get_num_nodes))
    for dst in line_topo.nodes():
        tm[0, dst] = 10
    tm[0, 0] = 0

    ratio, routing = optimal_load_balancing_LP_solver(line_topo, tm)
    assert ratio == line_topo.get_num_nodes - 1
    assert np.sum(routing[(0, 1)]) == line_topo.get_num_nodes - 1
    assert np.sum(routing[(1, 2)]) == line_topo.get_num_nodes - 2
    assert np.sum(routing[(2, 3)]) == line_topo.get_num_nodes - 3
    assert np.sum(routing[(3, 4)]) == line_topo.get_num_nodes - 4
    assert np.sum(routing[(4, 5)]) == line_topo.get_num_nodes - 5


def test_triangle_topology():
    triangle_topo = NetworkClass(BASIC_TOPOLOGIES["TRIANGLE"])
    tm = np.array([[0, 5, 10], [0, 0, 7], [0, 0, 0]])
    ratio, routing = optimal_load_balancing_LP_solver(triangle_topo, tm)
    assert ratio == 0.75
    assert (routing[(0, 1)] == np.array([[0, 1, 0.25], [0, 0, 0, ], [0, 0, 0, ]])).all()
    assert (routing[(0, 2)] == np.array([[0, 0, 0.75], [0, 0, 0, ], [0, 0, 0, ]])).all()
    assert (routing[(1, 2)] == np.array([[0, 0, 0.25], [0, 0, 1, ], [0, 0, 0, ]])).all()
