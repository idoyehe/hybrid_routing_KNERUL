"""
Created on 29 Jun 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""

from common.RL_Env.rl_env_consts import EnvConsts
from common.RL_Env.optimizer_abstract import *


class SoftMinOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, oblivious_routing_per_edge, testing=False):
        super(SoftMinOptimizer, self).__init__(net, testing)
        self._oblivious_routing_per_edge = oblivious_routing_per_edge

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, \
        total_load_per_link = self._get_cost_given_weights(weights_vector, traffic_matrix, optimal_value)

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, total_load_per_link

    def __get_edge_cost(self, cost_adj, each_edge_weight):
        cost_to_dst1 = cost_adj * self._graph_adjacency_matrix + each_edge_weight
        cost_to_dst2 = np.reshape(cost_to_dst1, [-1])
        cost_to_dst3 = cost_to_dst2[cost_to_dst2 != 0]
        return cost_to_dst3 * self._outgoing_edges

    def __soft_min(self, weights_vector, alpha=EnvConsts.SOFTMIN_ALPHA):
        """
        :param weights_vector: vector of weights
        :param alpha: for exponent expression
        :return: sum over deges
        """

        exp_val = np.exp(alpha * weights_vector)
        exp_val[weights_vector == 0] = 0
        exp_val[np.logical_and(weights_vector != 0, exp_val == 0)] = EnvConsts.EPSILON

        exp_val = np.transpose(exp_val) / np.sum(exp_val, axis=1)
        exp_val = np.sum(np.transpose(exp_val), axis=0)
        net_direct = self._network
        for u in net_direct.nodes:
            error_bound(1.0, sum(exp_val[net_direct.get_edge2id(u, v)] for _, v in net_direct.out_edges_by_node(u)))
        return exp_val

    def calculating_destination_based_spr(self, weights_vector):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network
        dst_splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        one_hop_cost = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)
        reduced_weighted_graph = self._build_reduced_weighted_graph(weights_vector)

        for node_dst in net_direct.nodes:
            cost_adj = nx.shortest_path_length(G=reduced_weighted_graph, target=node_dst, weight=EdgeConsts.WEIGHT_STR)
            cost_adj = [cost_adj[i] for i in net_direct.nodes]
            edge_cost = self.__get_edge_cost(cost_adj, one_hop_cost)
            q_val = self.__soft_min(edge_cost)
            for u, v in net_direct.edges:
                if u == node_dst:
                    continue
                edge_idx = net_direct.get_edge2id(u, v)
                dst_splitting_ratios[node_dst][u, v] = q_val[edge_idx]
        return dst_splitting_ratios

    def _get_cost_given_weights(self, weights_vector, traffic_matrix, optimal_value):
        dst_splitting_ratios = self.calculating_destination_based_spr(weights_vector)

        max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, \
        load_per_link = self._calculating_traffic_distribution(dst_splitting_ratios, traffic_matrix)
        if self._testing:
            logger.info("RL most congested link: {}".format(most_congested_link))
            logger.info("RL MLU: {}".format(max_congestion))
            logger.info("RL MLU Vs. Optimal: {}".format(max_congestion / optimal_value))

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link
