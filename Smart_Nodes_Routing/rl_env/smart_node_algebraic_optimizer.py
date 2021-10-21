from Learning_to_Route.rl_softmin_history.soft_min_optimizer import SoftMinOptimizer
from common.RL_Envs.optimizer_abstract import *
from common.utils import extract_flows
from common.consts import EdgeConsts
import numpy.linalg as npl


class SmartNodesOptimizer(SoftMinOptimizer):
    def __init__(self, net: NetworkClass, testing=False):
        super(SmartNodesOptimizer, self).__init__(net, -1, testing)

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

    def _get_cost_given_weights(self, links_weights, traffic_matrix, optimal_value):
        net_direct = self._network
        dst_splitting_ratios = self.calculating_destination_based_spr(links_weights)

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
            load_per_link = super(SmartNodesOptimizer, self)._calculating_traffic_distribution(dst_splitting_ratios, traffic_matrix)

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
