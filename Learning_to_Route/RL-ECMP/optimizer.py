"""
Created on 29 Jun 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""

from consts import HistoryConsts
from ecmp_network import *
from logger import logger


class WNumpyOptimizer:
    """TODO understand"""

    def __init__(self, net: ECMPNetwork, max_iterations=500):
        """
        constructor
        @param graph_adjacency_matrix: the graph adjacency matrix
        @param edges_capacities: all edges capacities
        @param max_iterations: number of max iterations
        """
        self._ecmp_network = net
        self._max_iterations = max_iterations
        self._nx_graph = True
        self._graph_adjacency_matrix = self._ecmp_network.get_adjacency
        self._edges_capacities = self._ecmp_network.get_edges_capacities()
        self._num_nodes = self._ecmp_network.get_num_nodes
        self._initialize()

    def _initialize(self):
        logger.debug("Building ingoing and outgoing edges map")
        _, self._ingoing_edges, self._outgoing_edges, _ = self._ecmp_network.build_edges_map()

        self._mask = np.ones((self._num_nodes, self._num_nodes), dtype=np.float32) - np.eye(self._num_nodes, dtype=np.float32)
        self._eye_masks = [np.expand_dims(self._mask[:, i], 1) for i in range(self._num_nodes)]

    def step(self, weights_vector, traffic_matrix):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        cost, congestion = self._get_cost_given_weights(weights_vector, traffic_matrix)
        return cost

    def _get_cost_given_weights(self, weights_vector, traffic_matrix):
        logger.debug("Calculate each edge weight")
        each_edge_weight = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)
        # calculating each edge it's weights in matrix |edges|x|edges|

        logger.debug("Creating directed graph based on edge weights")
        directed_graph = nx.from_numpy_matrix(each_edge_weight, create_using=nx.DiGraph())
        all_shortest_paths_costs = dict(nx.shortest_path_length(directed_graph, weight=EdgeConsts.WEIGHT_STR))
        # calculating shortest paths form each node to each node, dictionary way

        result = np.zeros_like(list(self._edges_capacities.values()), dtype=np.float32)
        for dst in range(len(traffic_matrix)):
            logger.info("Handle destination: {}".format(dst))
            dst_demand = np.expand_dims(traffic_matrix[:, dst], 1)  # getting all flows demands to dest from al sources
            cost_to_dest = [all_shortest_paths_costs[j][dst] for j in range(self._num_nodes)]  # getting all costs to dest from al sources

            edge_cost = self.__get_edge_cost(cost_to_dest, each_edge_weight)

            soft_min_cost_vector = self._soft_min(edge_cost)

            cong = self._get_flow_input_vector(soft_min_cost_vector, dst_demand, self._eye_masks[dst], dst)
            result += np.reshape(cong, [-1])  # , q_val

        congestion = result / list(self._edges_capacities.values())
        cost = np.max(congestion)
        return cost, congestion

    def __get_edge_cost(self, cost_to_dest, each_edge_weight):
        cost_to_dst1 = cost_to_dest * self._graph_adjacency_matrix + each_edge_weight
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

    def _get_flow_input_vector(self, softmin_cost_vector, demand, mask, dst):
        """
        input:
            demand: the demand of node
            q: is the softmin cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        """

        def flow_completed(prev):
            return (prev[dst] / demand_sum)[0]

        logger.debug("Begin simulate flow to: {}".format(dst))
        prev = self._get_new_val(softmin_cost_vector, demand, demand)
        prev_prev = demand
        cur_iter = 0
        demand_sum = np.sum(demand)

        while flow_completed(prev) < HistoryConsts.PERC_DEMAND:
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


# from Learning_to_Route.data_generation import tm_generation
# from topologies import topologies
# from Learning_to_Route.common.consts import Consts
#
# ecmpNetwork = ECMPNetwork(topologies["TRIANGLE"])
#
# opt = WNumpyOptimizer(ecmpNetwork)
# tm = tm_generation.one_sample_tm_base(ecmpNetwork, 1, Consts.GRAVITY, 0, 0, 0)
# opt.step(np.array([1, 1, 1, 1, 1, 1]), tm)
