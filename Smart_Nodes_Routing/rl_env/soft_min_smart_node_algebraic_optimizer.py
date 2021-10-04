from Learning_to_Route.rl_softmin_history.soft_min_optimizer import *
from common.RL_Envs.optimizer_abstract import *
from common.utils import extract_flows
from common.consts import EdgeConsts
import numpy.linalg as npl


class SoftMinSmartNodesOptimizer(SoftMinOptimizer):
    def __init__(self, net: NetworkClass, softMin_gamma=EnvConsts.SOFTMIN_GAMMA, testing=False):
        super(SoftMinSmartNodesOptimizer, self).__init__(net, softMin_gamma, testing)

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link = self._get_cost_given_weights(weights_vector,
                                                                                                                                       traffic_matrix,
                                                                                                                                       optimal_value)

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link

    def key_player_problem_comm_iter(self, weights_vector, k):
        net_direct = self._network
        kp_set = set()
        reduced_weighted_graph = self._build_reduced_weighted_graph(weights_vector)
        nodes_set = set(filter(lambda n: len(net_direct.out_edges_by_node(n)) > 1, net_direct.nodes))
        for _ in range(k):
            max_gbc = 0
            new_node = None
            for curr_node in nodes_set:
                t = nx.group_betweenness_centrality(reduced_weighted_graph, kp_set.union({curr_node}), normalized=True,
                                                    weight=EdgeConsts.WEIGHT_STR)
                if t > max_gbc:
                    max_gbc = t
                    new_node = curr_node
            kp_set.update({new_node})
            nodes_set.remove(new_node)

        return tuple(kp_set)

    def calculating_src_dst_spr(self, dst_splitting_ratios):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network
        smart_nodes_spr = net_direct.get_smart_nodes_spr
        src_dst_splitting_ratios = dict()
        for src, dst in net_direct.get_all_pairs():
            src_dst_splitting_ratios[(src, dst)] = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes),
                                                            dtype=np.float64)
            for node in net_direct.nodes:
                if node == dst:
                    continue
                for u, v in net_direct.out_edges_by_node(node):
                    assert u == node
                    # check whether smart node spt is exist otherwise return the default destination based
                    src_dst_splitting_ratios[(src, dst)][u, v] = smart_nodes_spr.get((src, dst, u, v), dst_splitting_ratios[dst, u, v])

        return src_dst_splitting_ratios

    def _get_cost_given_weights(self, weights_vector, traffic_matrix, optimal_value):
        net_direct = self._network
        dst_splitting_ratios = self.calculating_destination_based_spr(weights_vector)

        if len(net_direct.get_smart_nodes) > 0:
            src_dst_splitting_ratios = self.calculating_src_dst_spr(dst_splitting_ratios)
            max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, \
            load_per_link = self._calculating_traffic_distribution(src_dst_splitting_ratios, traffic_matrix)
        else:
            assert len(net_direct.get_smart_nodes) == 0
            max_congestion, \
            most_congested_link, \
            flows_to_dest_per_node, \
            congestion_per_link, \
            load_per_link = super(SoftMinSmartNodesOptimizer, self)._calculating_traffic_distribution(dst_splitting_ratios, traffic_matrix)

        if self._testing:
            logger.info("RL most congested link: {}".format(most_congested_link))
            logger.info("RL MLU: {}".format(max_congestion))
            logger.info("RL MLU Vs. Optimal: {}".format(max_congestion / optimal_value))

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link

    def _calculating_traffic_distribution(self, src_dst_splitting_ratios, tm):
        net_direct = self._network
        flows = extract_flows(tm)

        flows_src2dest_per_node = dict()
        for src, dst in flows:
            psi = src_dst_splitting_ratios[(src, dst)]
            demand = np.zeros(shape=(net_direct.get_num_nodes))
            demand[src] = tm[src, dst]
            assert all(psi[dst][:] == 0)
            assert psi.shape == (net_direct.get_num_nodes, net_direct.get_num_nodes)
            flows_src2dest_per_node[(src, dst)] = demand @ npl.inv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)

        if logger.level == logging.DEBUG:
            self.__validate_flow(net_direct, tm, flows_src2dest_per_node, src_dst_splitting_ratios)

        total_load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            edge_index = net_direct.get_edge2id(u, v)
            total_load_per_link[edge_index] = sum(
                flows_src2dest_per_node[(src, dst)][u] * src_dst_splitting_ratios[src, dst][u, v] for src, dst in flows)

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]

        return max_congestion, most_congested_link, flows_src2dest_per_node, total_congestion_per_link, total_load_per_link

    @staticmethod
    def __validate_flow(net_direct: NetworkClass, tm, flows_src2dest_per_node, src_dst_splitting_ratios):
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

    def calculating_effective_betweenness(self, weights_vector):
        net_direct = self._network
        dst_splitting_ratios = self.calculating_destination_based_spr(weights_vector)
        b = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes))
        for k in net_direct.nodes:
            psi = dst_splitting_ratios[k]
            b += npl.inv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi) @ psi

        B = np.zeros(shape=(net_direct.get_num_nodes))
        for j in net_direct.nodes:
            B[j] = np.sum(b[:, j])

        return B
