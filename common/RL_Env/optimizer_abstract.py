from common.logger import *
from common.network_class import NetworkClass, nx
import numpy as np
import numpy.linalg as npl
from common.utils import error_bound
from common.consts import EdgeConsts


class Optimizer_Abstract(object):
    def __init__(self, net: NetworkClass, testing=False):
        """
        constructor
        @param graph_adjacency_matrix: the graph adjacency matrix
        @param edges_capacities: all edges capacities
        @param max_iterations: number of max iterations
        """
        self._network = net
        self._graph_adjacency_matrix = self._network.get_adjacency
        self._num_nodes = self._network.get_num_nodes
        self._num_edges = self._network.get_num_edges
        self._initialize()
        self._testing = testing

    def _initialize(self):
        logger.debug("Building ingoing and outgoing edges map")
        self._ingoing_edges, self._outgoing_edges, self._edges_capacities = self._network.build_edges_map()

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        pass

    @staticmethod
    def __validate_flow(net_direct, traffic_matrix, flows_dest_per_node, splitting_ratios):
        for dst in net_direct.nodes:
            current_spr = splitting_ratios[dst]
            assert error_bound(flows_dest_per_node[dst][dst], sum(traffic_matrix[:, dst]))
            for node in net_direct.nodes:
                assert flows_dest_per_node[dst][node] >= traffic_matrix[node, dst]
                _flow_to_node = sum(
                    flows_dest_per_node[dst][u] * current_spr[u, v] if u != dst else 0 for u, v in
                    net_direct.in_edges_by_node(node))
                _flow_from_node = sum(
                    flows_dest_per_node[dst][u] * current_spr[u, v] if u != dst else 0 for u, v in
                    net_direct.out_edges_by_node(node))

                if node == dst:
                    assert error_bound(_flow_from_node, 0)
                else:
                    assert error_bound(_flow_to_node + traffic_matrix[node, dst], _flow_from_node)

    def _calculating_traffic_distribution(self, dst_splitting_ratios, traffic_matrix):
        net_direct = self._network
        flows_to_dest_per_node = dict()
        for dst in net_direct.nodes:
            psi = dst_splitting_ratios[dst]
            demands = traffic_matrix[:, dst]
            assert all(psi[dst][:] == 0)
            assert psi.shape == (net_direct.get_num_nodes, net_direct.get_num_nodes)
            A = np.transpose(np.identity(net_direct.get_num_nodes) - psi)
            flows_to_dest_per_node[dst] = npl.solve(A, demands)

        self.__validate_flow(net_direct, traffic_matrix, flows_to_dest_per_node, dst_splitting_ratios)

        load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            edge_index = net_direct.get_edge2id(u, v)
            load_per_link[edge_index] = sum(flows_to_dest_per_node[dst][u] * dst_splitting_ratios[dst][u, v] for dst in net_direct.nodes)

        congestion_per_link = load_per_link / self._edges_capacities

        most_congested_link = np.argmax(congestion_per_link)
        max_congestion = congestion_per_link[most_congested_link]

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link

    def _build_reduced_weighted_graph(self, weights_vector):
        net_direct = self._network
        reduced_weighted_graph = nx.DiGraph()
        for edge_index, edge_weight in enumerate(weights_vector):
            u, v = net_direct.get_id2edge(edge_index)
            reduced_weighted_graph.add_edge(u, v, **{EdgeConsts.WEIGHT_STR: edge_weight})

        return reduced_weighted_graph

    def _get_cost_given_weights(self, weights_vector, traffic_matrix, optimal_value):
        pass

    def calculating_destination_based_spr(self, weights_vector):
        pass
