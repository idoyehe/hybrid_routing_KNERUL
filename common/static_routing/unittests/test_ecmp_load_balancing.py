import numpy as np
from topologies import BASIC_TOPOLOGIES
from ecmp_load_balancing import ecmp_arch_congestion
from network_class import NetworkClass


def test_star_topology():
    star_topo = NetworkClass(BASIC_TOPOLOGIES["STAR"])
    tm = np.zeros(shape=(star_topo.get_num_nodes, star_topo.get_num_nodes))
    for dst in star_topo.nodes():
        tm[0, dst] = 10
    tm[0, 0] = 0
    routing = ecmp_arch_congestion(star_topo, tm)
    for edge, edge_routing in routing.items():
        assert edge_routing == 10


def test_ring_topology():
    ring_topo = NetworkClass(BASIC_TOPOLOGIES["RING"])
    tm = np.zeros(shape=(ring_topo.get_num_nodes, ring_topo.get_num_nodes))
    for node in ring_topo.nodes():
        tm[node, (node + 1) % ring_topo.get_num_nodes] = 10
    routing = ecmp_arch_congestion(ring_topo, tm)
    for edge, edge_routing in routing.items():
        assert edge_routing == 10


def test_fcn_topology():
    fcn_topo = NetworkClass(BASIC_TOPOLOGIES["FCN"])
    tm = np.zeros(shape=(fcn_topo.get_num_nodes, fcn_topo.get_num_nodes))
    for src, dst in fcn_topo.get_all_pairs():
        tm[src, dst] = 10

    routing = ecmp_arch_congestion(fcn_topo, tm)
    for edge, edge_routing in routing.items():
        assert edge_routing == 10


def test_tree_topology():
    tree_topo = NetworkClass(BASIC_TOPOLOGIES["TREE"])
    tm = np.zeros(shape=(tree_topo.get_num_nodes, tree_topo.get_num_nodes))
    for dst in tree_topo.nodes():
        tm[0, dst] = 10
    tm[0, 0] = 0

    routing = ecmp_arch_congestion(tree_topo, tm)
    assert np.sum(routing[(0, 1)]) == 20
    assert np.sum(routing[(0, 2)]) == 30


def test_line_topology():
    line_topo = NetworkClass(BASIC_TOPOLOGIES["LINE"])
    tm = np.zeros(shape=(line_topo.get_num_nodes, line_topo.get_num_nodes))
    for dst in line_topo.nodes():
        tm[0, dst] = 10
    tm[0, 0] = 0

    routing = ecmp_arch_congestion(line_topo, tm)
    assert np.sum(routing[(0, 1)]) == (line_topo.get_num_nodes - 1) * 10
    assert np.sum(routing[(1, 2)]) == (line_topo.get_num_nodes - 2) * 10
    assert np.sum(routing[(2, 3)]) == (line_topo.get_num_nodes - 3) * 10
    assert np.sum(routing[(3, 4)]) == (line_topo.get_num_nodes - 4) * 10
    assert np.sum(routing[(4, 5)]) == (line_topo.get_num_nodes - 5) * 10
