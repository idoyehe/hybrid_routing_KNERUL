"""
Created on 21 Dec 2020
@author: Ido Yehezkel
"""

from common.rl_env_consts import HistoryConsts
from common.network_class import *
from common.logger import logger
from static_routing.oblivious_routing import oblivious_routing, calculate_congestion_per_matrices


class WNumpyOptimizer_Oblivious:
    def __init__(self, net: NetworkClass, max_iterations=500, testing=False):
        """
        constructor
        @param graph_adjacency_matrix: the graph adjacency matrix
        @param edges_capacities: all edges capacities
        @param max_iterations: number of max iterations
        """
        self._network = net
        self._max_iterations = max_iterations
        self._nx_graph = True
        self._testing = testing
        self._graph_adjacency_matrix = self._network.get_adjacency
        self._num_nodes = self._network.get_num_nodes
        self._initialize()
        self._max_iters = 500
        self.oblivious_ratio, self.oblivious_routing_per_edge, self.per_flow_routing_scheme = oblivious_routing(
            self._network)

        self.rl_vs_obliv_data = None
        self.obliv_congestion_ratio = None

    def _initialize(self):
        logger.debug("Building ingoing and outgoing edges map")
        _, self._ingoing_edges, self._outgoing_edges, self._edges_capacities = self._network.build_edges_map()

        self._mask = np.ones((self._num_nodes, self._num_nodes), dtype=np.float32) - np.eye(self._num_nodes,
                                                                                            dtype=np.float32)
        self._eye_masks = [np.expand_dims(self._mask[:, i], 1) for i in range(self._num_nodes)]
        self._zero_diagonal = np.ones_like(self._graph_adjacency_matrix, dtype=np.float32) - np.eye(self._num_nodes,
                                                                                                    dtype=np.float32)

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        cost = self._get_cost_given_weights(weights_vector, traffic_matrix, optimal_value)
        return cost

    def _set_cost_to_dsts(self, weights_vector):
        tmp = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)
        tmp[tmp == 0] = HistoryConsts.INFTY
        return tmp * self._zero_diagonal

    def _run_destination_demands(self, q_val, dst_demand, mask, dst=None):
        '''
        input:
            v: the node with demands
            demand: the demand of node v
            q: is the softmax cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        '''

        # loop magic goes here, basically converts the for loop into matrix operations
        def get_new_val(prev_iteration, current_demands):
            current_demand_each_node = self._ingoing_edges @ (
                    np.transpose(q_val * self._outgoing_edges) @ current_demands)
            return prev_iteration + current_demand_each_node

        prev_iteration = dst_demand
        current_iteration = get_new_val(dst_demand, dst_demand)
        cur_iter = 1
        total_demand = np.sum(dst_demand)

        def does_demand_completed(_prev):
            return (np.sum(_prev * (1 - mask)) / total_demand) < HistoryConsts.PERC_DEMAND

        while does_demand_completed(current_iteration):
            res_diff = (current_iteration - prev_iteration) * mask
            tmp = get_new_val(current_iteration, res_diff)
            prev_iteration = current_iteration
            current_iteration = tmp
            if cur_iter == self._max_iters:
                break
            cur_iter += 1

        final_s_value = current_iteration
        edge_congestion = np.sum(np.transpose(q_val) @ (final_s_value * mask), axis=1)
        return edge_congestion  # final_s_value,

    def _get_cost_given_weights(self, weights_vector, traffic_matrix, optimal_value):
        logger.debug("Calculate each edge weight")
        one_hop_cost = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)
        reduced_directed_graph = nx.from_numpy_matrix(one_hop_cost, create_using=nx.DiGraph())
        cost_all_adj = dict(nx.shortest_path_length(reduced_directed_graph, weight='weight'))
        rl_total_load_per_arch = np.zeros_like(weights_vector, dtype=np.float64)
        obliv_total_load_per_arch = np.zeros_like(weights_vector, dtype=np.float64)

        self.obliv_congestion_ratio, self._obliv_total_load_per_arch, _ = calculate_congestion_per_matrices(
            self._network, [(traffic_matrix, optimal_value)], self.oblivious_routing_per_edge)

        for index in range(len(obliv_total_load_per_arch)):
            src, dst = self._network.get_id2edge()[index]
            obliv_total_load_per_arch[index] = self._obliv_total_load_per_arch[src, dst]

        oblivious_most_congested = np.argmax(obliv_total_load_per_arch)
        oblivious_most_congested_arch = self._network.get_id2edge()[oblivious_most_congested]

        for node_dst in self._network.nodes:
            dest_demands = np.array(traffic_matrix[:, node_dst])
            total_demands = np.sum(dest_demands)
            if total_demands == 0.0:
                continue

            cost_adj = [cost_all_adj[i][node_dst] for i in range(self._num_nodes)]

            edge_cost = self.__get_edge_cost(cost_adj, one_hop_cost)

            q_val = self._soft_min(edge_cost)
            dest_demands = np.expand_dims(traffic_matrix[:, node_dst], 1)
            loads = self._run_destination_demands(q_val, dest_demands, self._eye_masks[node_dst])

            rl_total_load_per_arch += loads

        congestion_per_link = rl_total_load_per_arch / self._edges_capacities
        most_congested_arch = np.argmax(congestion_per_link)
        max_congestion_ratio = congestion_per_link[most_congested_arch] / optimal_value
        most_congested_arch = self._network.get_id2edge()[most_congested_arch]
        if self._testing:
            print("RL: most_congested_arch: {} congestion ratio: {}".format(most_congested_arch, max_congestion_ratio))
            print("Oblivious: most_congested_arch: {} congestion ratio: {}".format(oblivious_most_congested_arch,
                                                                                   self.obliv_congestion_ratio[0]))

            print("RL Vs. Oblivious: {} ".format(max_congestion_ratio / self.obliv_congestion_ratio[0]))
        cost = np.abs(obliv_total_load_per_arch - rl_total_load_per_arch) / self._edges_capacities
        return np.sum(cost)

    def __get_edge_cost(self, cost_adj, each_edge_weight):
        cost_to_dst1 = cost_adj * self._graph_adjacency_matrix + each_edge_weight
        cost_to_dst2 = np.reshape(cost_to_dst1, [-1])
        cost_to_dst3 = cost_to_dst2[cost_to_dst2 != 0]
        return cost_to_dst3 * self._outgoing_edges

    @staticmethod
    def _soft_min(weights_vector, alpha=HistoryConsts.SOFTMIN_ALPHA):
        """
        :param weights_vector: vector of weights
        :param alpha: for exponent expression
        :return: sum over deges
        """

        exp_val = np.exp(alpha * weights_vector)
        exp_val[weights_vector == 0] = 0
        exp_val[np.logical_and(weights_vector != 0, exp_val == 0)] = HistoryConsts.EPSILON

        exp_val = np.transpose(exp_val) / np.sum(exp_val, axis=1)

        return np.transpose(exp_val)  # sum over edges

    def _get_new_val(self, softmin_cost_vector, prev_val, mul_val):
        # loop magic goes here, basically converts the for loop into matrix operations
        return prev_val + self._ingoing_edges @ (np.transpose(softmin_cost_vector * self._outgoing_edges) @ mul_val)

    def _get_flow_input_vector(self, softmin_cost_vector, demand, total_demands, mask, dst):
        """
        input:
            demand: the demand of node
            q: is the softmin cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        """

        logger.debug("Begin simulate flow to: {}".format(dst))
        prev = self._get_new_val(softmin_cost_vector, demand, demand)
        prev_prev = demand
        cur_iter = 0

        assert total_demands > 0.0

        while float(prev[dst]) / total_demands < HistoryConsts.PERC_DEMAND:
            logger.debug("Iteration #: {}".format(cur_iter))
            res_diff = (prev - prev_prev) * mask
            tmp = self._get_new_val(softmin_cost_vector, prev, res_diff)
            prev_prev = prev
            prev = tmp
            if cur_iter == self._max_iterations:
                break
            cur_iter += 1

        final_s_value = prev
        edge_congestion = np.sum(np.transpose(softmin_cost_vector) @ (final_s_value * mask), axis=1)
        return edge_congestion  # final_s_value
