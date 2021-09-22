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

    def __get_distance_via_neighbor(self, cost_adj, each_edge_weight):
        distance_via_neighbor = cost_adj * self._graph_adjacency_matrix
        distance_via_neighbor[self._graph_adjacency_matrix == 0] = np.inf
        distance_via_neighbor += each_edge_weight
        return distance_via_neighbor

    def __soft_min(self, dest, distance_via_neighbor, gamma=EnvConsts.SOFTMIN_ALPHA):
        exp_val = np.exp(gamma * distance_via_neighbor)
        normalizer = np.sum(exp_val, axis=1)
        exp_val = np.transpose(np.transpose(exp_val) / normalizer)
        exp_val[dest, :] = 0.0
        assert all(error_bound(int(u != dest), sum(exp_val[u])) for u in self._network.nodes)
        return exp_val

    def calculating_destination_based_spr(self, weights_vector):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network
        splitting_ratios_per_dest = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        one_hop_cost = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)
        reduced_weighted_graph = self._build_reduced_weighted_graph(weights_vector)

        for dest in net_direct.nodes:
            shortest_path2dest = nx.shortest_path_length(G=reduced_weighted_graph, target=dest, weight=EdgeConsts.WEIGHT_STR)
            shortest_path2dest = np.array([shortest_path2dest[i] for i in net_direct.nodes], dtype=np.float64)
            distance_via_neighbor = self.__get_distance_via_neighbor(shortest_path2dest, one_hop_cost)
            splitting_ratios_per_dest[dest] = self.__soft_min(dest, distance_via_neighbor)
        return splitting_ratios_per_dest

    def _get_cost_given_weights(self, weights_vector, traffic_matrix, optimal_value):
        dst_splitting_ratios = self.calculating_destination_based_spr(weights_vector)

        max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, \
        load_per_link = self._calculating_traffic_distribution(dst_splitting_ratios, traffic_matrix)
        if self._testing:
            logger.info("RL most congested link: {}".format(most_congested_link))
            logger.info("RL MLU: {}".format(max_congestion))
            logger.info("RL MLU Vs. Optimal: {}".format(max_congestion / optimal_value))

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link
