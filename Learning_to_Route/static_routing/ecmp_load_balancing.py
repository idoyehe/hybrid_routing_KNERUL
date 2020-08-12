from common.consts import EdgeConsts
from common.network_class import NetworkClass, nx
from collections import defaultdict
from common.logger import *
from Learning_to_Route.common.utils import extract_flows
from itertools import tee
import numpy as np


def ecmp_arch_congestion(net: NetworkClass, traffic_matrix, weight=None):
    per_edge_flow_fraction = defaultdict(int)
    assert net.g_is_directed
    if weight is not None:
        logger.info("ECMP by weight label is: {}".format(weight))
    else:
        logger.info("ECMP by shortest length paths")

    flows = extract_flows(traffic_matrix)
    for flow in flows:
        src, dst = flow
        assert src != dst
        assert traffic_matrix[flow] > 0
        logger.debug("ECMP of flow: {} -> {}".format(src, dst))
        shortest_path_generator = net.all_shortest_path(source=src, target=dst, weight=weight)
        shortest_path_generator, number_of_paths = tee(shortest_path_generator, 2)
        number_of_paths = sum(1.0 for _ in number_of_paths)
        flow_fraction: float = traffic_matrix[flow] / number_of_paths

        for _path in shortest_path_generator:
            for _arch in list(list(map(nx.utils.pairwise, [_path]))[0]):
                if isinstance(net.get_graph, nx.MultiDiGraph):
                    duplicate_archs = list(filter(lambda e: e[0] == _arch[0] and e[1] == _arch[1], net.edges))
                    number_of_duplicate_archs = len(duplicate_archs)
                    for _arch in duplicate_archs:
                        logger.debug("Handle edge {} in path {}".format(str(_arch), str(_path)))
                        per_edge_flow_fraction[_arch] += flow_fraction / number_of_duplicate_archs
                else:
                    logger.debug("Handle edge {} in path {}".format(str(_arch), str(_path)))
                    per_edge_flow_fraction[_arch] += flow_fraction

    return per_edge_flow_fraction


if __name__ == "__main__":
    def get_base_graph():
        # init a triangle if we don't get a network graph
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 100}),
                          (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 100}),
                          (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 150})])

        return g


    def get_flows_matrix():
        return np.array([[0, 50, 100], [0, 0, 70], [50, 0, 0]])


    net = NetworkClass(get_base_graph())

    per_edge_ecmp_flow_fraction = ecmp_arch_congestion(net.get_g_directed, get_flows_matrix())
    assert per_edge_ecmp_flow_fraction[(0, 1)] == 50
    assert per_edge_ecmp_flow_fraction[(0, 2)] == 100
    assert per_edge_ecmp_flow_fraction[(1, 2)] == 70
    assert per_edge_ecmp_flow_fraction[(2, 0)] == 50
