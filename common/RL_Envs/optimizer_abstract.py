import warnings
from common.logger import *
from common.network_class import NetworkClass, nx
import numpy as np
import numpy.linalg as npl
from common.utils import error_bound, extract_flows
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
    def __validate_flow(net_direct, traffic_matrix, flows_dest_per_node, active_dest, splitting_ratios):
        for dst in net_direct.nodes:
            if np.sum(traffic_matrix[:, dst]) > 0:
                current_spr = splitting_ratios[dst]
                assert error_bound(flows_dest_per_node[dst][dst], sum(traffic_matrix[:, dst]))
                for node in net_direct.nodes:
                    assert flows_dest_per_node[dst][node] >= traffic_matrix[node, dst] or error_bound(flows_dest_per_node[dst][node],
                                                                                                      traffic_matrix[node, dst])
                    _flow_to_node = sum(
                        flows_dest_per_node[dst][u] * current_spr[u, v] for u, v in net_direct.in_edges_by_node(node))
                    _flow_from_node = sum(
                        flows_dest_per_node[dst][u] * current_spr[u, v] for u, v in net_direct.out_edges_by_node(node))

                    if node == dst:
                        assert error_bound(_flow_from_node, 0)
                    else:
                        assert error_bound(_flow_to_node + traffic_matrix[node, dst], _flow_from_node)

    def _calculating_traffic_distribution(self, dst_splitting_ratios, traffic_matrix):
        net_direct = self._network
        flows_to_dest_per_node = dict()
        active_dest = tuple(filter(lambda dst: np.sum(traffic_matrix[:, dst]) > 0, net_direct.nodes))
        for dst in active_dest:
            psi = dst_splitting_ratios[dst]
            demands = traffic_matrix[:, dst]
            assert all(psi[dst][:] == 0)
            assert psi.shape == (net_direct.get_num_nodes, net_direct.get_num_nodes)

            try:
                flows_to_dest_per_node[dst] = demands @ npl.inv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)
            except npl.LinAlgError as e:
                warnings.warn(str(e))
                flows_to_dest_per_node[dst] = demands @ npl.pinv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)

        self.__validate_flow(net_direct, traffic_matrix, flows_to_dest_per_node, active_dest, dst_splitting_ratios)

        load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            edge_index = net_direct.get_edge2id(u, v)
            load_per_link[edge_index] = sum(flows_to_dest_per_node[dst][u] * dst_splitting_ratios[dst][u, v] for dst in active_dest)

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

    def calculating_src_dst_spr(self, dst_splitting_ratios):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network
        smart_nodes_spr = net_direct.get_smart_nodes_spr
        src_dst_splitting_ratios = dict()
        for src, dst in net_direct.get_all_pairs():
            src_dst_splitting_ratios[(src, dst)] = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
            for node in net_direct.nodes:
                if node == dst:
                    continue
                for u, v in net_direct.out_edges_by_node(node):
                    assert u == node
                    # check whether smart node spt is exist otherwise return the default destination based
                    src_dst_splitting_ratios[(src, dst)][u, v] = np.float64(smart_nodes_spr.get((src, dst, u, v), dst_splitting_ratios[dst, u, v]))

            assert all(error_bound(int(u != dst), sum(src_dst_splitting_ratios[(src, dst)][u])) for u in self._network.nodes)

        return src_dst_splitting_ratios

    @staticmethod
    def reduce_src_dst_spr_to_dst_spr(net: NetworkClass, src_dst_splitting_ratios):
        dst_splitting_ratios = np.zeros(shape=(net.get_num_nodes, net.get_num_nodes, net.get_num_nodes), dtype=np.float64)
        for t in net.nodes:
            relevant_flow = list(filter(lambda f: f[1] == t, net.get_all_pairs()))
            for u in net.nodes:
                non_zero = 0
                s = np.zeros(shape=(net.get_num_nodes), dtype=np.float64)
                for src, dst in relevant_flow:
                    if sum(src_dst_splitting_ratios[src, dst, u]) > 0:
                        s += src_dst_splitting_ratios[src, dst, u]
                        non_zero += 1
                if non_zero > 0:
                    s /= non_zero
                    dst_splitting_ratios[t][u] = s

        return dst_splitting_ratios

    @staticmethod
    def __validate_src_dst_flow(net_direct: NetworkClass, tm, flows_src2dest_per_node, src_dst_splitting_ratios):
        flows = extract_flows(tm)
        for src, dst in flows:
            current_spr = src_dst_splitting_ratios[src, dst]
            assert flows_src2dest_per_node[src, dst][src] > tm[src, dst] or error_bound(flows_src2dest_per_node[src, dst][src], tm[src, dst])
            assert error_bound(flows_src2dest_per_node[src, dst][dst], tm[src, dst])
            for node in net_direct.nodes:
                _flow_to_node = sum(
                    flows_src2dest_per_node[src, dst][u] * current_spr[u, v] if u != dst else 0 for u, v in
                    net_direct.in_edges_by_node(node))
                _flow_from_node = sum(
                    flows_src2dest_per_node[src, dst][u] * current_spr[u, v] if u != dst else 0 for u, v in
                    net_direct.out_edges_by_node(node))

                if node == src:
                    assert error_bound(flows_src2dest_per_node[src, dst][node], _flow_to_node + tm[src, dst])
                    assert error_bound(flows_src2dest_per_node[src, dst][node], _flow_from_node)
                elif node == dst:
                    assert error_bound(_flow_to_node, tm[src, dst])
                    assert error_bound(_flow_from_node, 0)
                else:
                    assert error_bound(flows_src2dest_per_node[src, dst][node], _flow_to_node)
                    assert error_bound(_flow_from_node, _flow_to_node)

    def _calculating_src_dst_traffic_distribution(self, src_dst_splitting_ratios, tm):
        net_direct = self._network
        flows = extract_flows(tm)

        flows_src2dest_per_node = dict()
        for src, dst in flows:
            psi = src_dst_splitting_ratios[(src, dst)]
            demand = np.zeros(shape=(net_direct.get_num_nodes))
            demand[src] = tm[src, dst]
            assert all(psi[dst][:] == 0)
            assert psi.shape == (net_direct.get_num_nodes, net_direct.get_num_nodes)
            try:
                flows_src2dest_per_node[(src, dst)] = demand @ npl.inv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)
            except:
                warnings.warn("Singular")
                flows_src2dest_per_node[(src, dst)] = demand @ npl.pinv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)


        if logger.level == logging.DEBUG:
            self.__validate_src_dst_flow(net_direct, tm, flows_src2dest_per_node, src_dst_splitting_ratios)

        total_load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            edge_index = net_direct.get_edge2id(u, v)
            total_load_per_link[edge_index] = sum(
                flows_src2dest_per_node[(src, dst)][u] * src_dst_splitting_ratios[src, dst][u, v] for src, dst in flows)

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]

        return max_congestion, most_congested_link, flows_src2dest_per_node, total_congestion_per_link, total_load_per_link
