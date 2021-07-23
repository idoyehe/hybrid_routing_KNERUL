"""
Created on 29 Jun 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""

from common.RL_Env.rl_env_consts import HistoryConsts
from common.static_routing.oblivious_routing import calculate_congestion_per_matrices, oblivious_routing
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
        rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, \
        rl_total_load_per_link = self._get_cost_given_weights(weights_vector, traffic_matrix, optimal_value)

        return rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, \
               rl_total_load_per_link

    def __get_edge_cost(self, cost_adj, each_edge_weight):
        cost_to_dst1 = cost_adj * self._graph_adjacency_matrix + each_edge_weight
        cost_to_dst2 = np.reshape(cost_to_dst1, [-1])
        cost_to_dst3 = cost_to_dst2[cost_to_dst2 != 0]
        return cost_to_dst3 * self._outgoing_edges

    def _soft_min(self, weights_vector, alpha=HistoryConsts.SOFTMIN_ALPHA):
        """
        :param weights_vector: vector of weights
        :param alpha: for exponent expression
        :return: sum over deges
        """

        exp_val = np.exp(alpha * weights_vector)
        exp_val[weights_vector == 0] = 0
        exp_val[np.logical_and(weights_vector != 0, exp_val == 0)] = HistoryConsts.EPSILON

        exp_val = np.transpose(exp_val) / np.sum(exp_val, axis=1)
        exp_val = np.sum(np.transpose(exp_val), axis=0)
        net_direct = self._network
        for u in net_direct.nodes:
            error_bound(1.0, sum(exp_val[net_direct.get_edge2id(u, v)] for _, v in net_direct.out_edges_by_node(u)))
        return exp_val

    def _get_cost_given_weights(self, weights_vector, tm, optimal_value):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network

        splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)
        one_hop_cost = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)

        reduced_directed_graph = nx.DiGraph()
        for edge_index, cost in enumerate(weights_vector):
            u, v = net_direct.get_id2edge(edge_index)
            reduced_directed_graph.add_edge(u, v, cost=cost)

        for node_dst in net_direct.nodes:
            cost_adj = nx.shortest_path_length(G=reduced_directed_graph, target=node_dst, weight='cost')
            cost_adj = [cost_adj[i] for i in range(self._num_nodes)]
            edge_cost = self.__get_edge_cost(cost_adj, one_hop_cost)
            q_val = self._soft_min(edge_cost)
            splitting_ratios[node_dst] = q_val

        rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, \
        rl_total_load_per_link = self._calculating_traffic_distribution(splitting_ratios, tm)

        if self._testing:
            self.vs_oblivious_data = None
            if self._oblivious_routing_per_edge is not None:
                oblv_congestion, obliv_rl_total_load_per_arch, oblv_congestion_link_histogram = \
                    calculate_congestion_per_matrices(self._network, [(tm, optimal_value)],
                                                      self._oblivious_routing_per_edge)
                oblv_congestion = oblv_congestion[0]

                assert np.sum(oblv_congestion_link_histogram) == 1
                oblv_most_congested_arch = str(self._network.get_id2edge(np.argmax(oblv_congestion_link_histogram)))
                rl_most_congested_arch = str(self._network.get_id2edge(rl_most_congested_link))
                print("Oblivious most congested link: {}".format(oblv_most_congested_arch))
                rl_oblivious_ratio = np.abs((rl_max_congestion / optimal_value) / oblv_congestion)

                self.vs_oblivious_data = \
                    (rl_most_congested_arch,
                     oblv_most_congested_arch,
                     (rl_most_congested_arch == oblv_most_congested_arch),
                     rl_oblivious_ratio)
                print("Oblivious cost value: {}".format(oblv_congestion))
                print("RL Oblivious Ratio: {}".format(rl_oblivious_ratio))

            print("RL most congested link: {}".format(rl_most_congested_link))
            print("RL cost value: {}".format(rl_max_congestion / optimal_value))

        return rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, \
               rl_total_load_per_link


if __name__ == "__main__":
    from common.topologies import BASIC_TOPOLOGIES
    from common.static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver

    network = NetworkClass(BASIC_TOPOLOGIES["TRIANGLE"])
    tm = np.array([[0, 10, 0], [0, 0, 0], [0, 0, 0]])
    oblivious_ratio, oblivious_routing_per_edge, per_flow_routing_scheme = oblivious_routing(network)
    opt = SoftMinOptimizer(network, oblivious_routing_per_edge)
    opt_congestion, necessary_capacity = optimal_load_balancing_LP_solver(net=network, traffic_matrix=tm)
    print("Optimal Congestion: {}".format(opt_congestion))
    max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
        opt.step([100, 100, 0.00000001, 100, 0.00000001, 0.00000001], tm, opt_congestion)
    print("Optimizer Congestion: {}".format(max_congestion))
    print("Congestion Ratio :{}".format(max_congestion / opt_congestion))
