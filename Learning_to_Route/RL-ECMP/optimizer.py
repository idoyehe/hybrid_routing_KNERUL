"""
Created on 29 Jun 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""

from consts import HistoryConsts
from copy import deepcopy
from history import utils
import numpy as np
import networkx as nx


class WNumpyOptimizer:
    """TODO understand"""

    def __init__(self, graph_adjacency_matrix, edges_capacities, max_iterations=500):
        """
        constructor
        @param graph_adjacency_matrix: the graph adjacency matrix
        @param edges_capacities: all edges capacities
        @param max_iterations: number of max iterations
        """
        self._max_iterations = max_iterations
        self._nx_graph = True
        self._graph_adjacency_matrix = graph_adjacency_matrix
        self._edges_capacities = edges_capacities
        self._num_nodes = self._graph_adjacency_matrix.shape[0]
        self.__initilize()

    def __initilize(self):
        self._num_edges, self._ingoing_edges, self._outgoing_edges, _ = utils.build_edges_map(self._graph_adjacency_matrix)

        self._zero_diagonal = np.ones_like(self._graph_adjacency_matrix, dtype=np.float32) - np.eye(self._num_nodes, dtype=np.float32)

        self._mask = np.ones((self._num_nodes, self._num_nodes), dtype=np.float32) - np.eye(self._num_nodes, dtype=np.float32)
        self._eye_mask = np.eye(self._num_nodes, dtype=np.float32)
        self._eye_masks = [np.expand_dims(self._mask[:, i], 1) for i in range(self._num_nodes)]
        self._mask = np.transpose(self._mask)

        self._q_zero_one = self._outgoing_edges.copy()
        self._nonzero_outgoing = np.nonzero(self._outgoing_edges)
        self._a_out_multi = np.tile(self._outgoing_edges, (1, self._num_nodes))

        self._routing_demand_sum = {k: np.ones((k, 1)) for k in range(1, self._num_nodes)}

        self._demand_for_dst = {}
        for dst in range(self._num_nodes):
            demand = np.eye(self._num_nodes)
            demand = np.delete(demand, dst, 0)
            demand = np.reshape(demand, (self._num_nodes - 1, self._num_nodes, 1))
            self._demand_for_dst[dst] = demand

        self._eye_nodes = np.eye(self._num_nodes)

    def step(self, w, traffic_matrix):
        """understand each of this functions"""
        cost, congestion = self._get_cost_given_weights(w, traffic_matrix)
        return -cost

    def __get_edge_cost(self, cost_to_dest, each_edge_weight):
        cost_to_dst1 = cost_to_dest * self._graph_adjacency_matrix + each_edge_weight
        cost_to_dst2 = np.reshape(cost_to_dst1, [-1])
        cost_to_dst3 = cost_to_dst2[cost_to_dst2 != 0]
        return cost_to_dst3 * self._outgoing_edges

    @staticmethod
    def _softmin(v, axis=1, alpha=HistoryConsts.SOFTMIN_ALPHA):
        # this is semi true, we need to take into account the in degree of things
        # as the softmax is vector dependet
        # if we assume v is a matrix this will help, and now we run softmin
        # across the per vector direction

        exp_val = np.exp(alpha * v)
        exp_val[v == 0] = 0
        exp_val[np.logical_and(v != 0, exp_val == 0)] = HistoryConsts.EPSILON

        exp_val = np.transpose(exp_val) / np.sum(exp_val, axis=1)

        return np.transpose(exp_val)  # sum over edges

    def _get_new_val(self, softmin_cost_vector, prev_val, mul_val):
        # loop magic goes here, basically converts the for loop into matrix operations
        return prev_val + self._ingoing_edges @ (np.transpose(softmin_cost_vector * self._outgoing_edges) @ mul_val)

    def _get_flow_input_vector(self, softmin_cost_vector, demand, mask, dst=None):
        """
        input:
            demand: the demand of node
            q: is the softmin cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        """

        def flow_completed(prev):
            if dst is None:
                return np.sum(prev * (1 - mask)) / demand_sum
            else:
                return prev[dst] / demand_sum

        prev = self._get_new_val(softmin_cost_vector, demand, demand)
        prev_prev = demand
        cur_iter = 0
        demand_sum = np.sum(demand)


        while flow_completed(prev) < HistoryConsts.PERC_DEMAND:
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

    def _get_cost_given_weights(self, w, demand_split):
        each_edge_weight = (w * self._outgoing_edges) @ np.transpose(self._ingoing_edges)
        cost_all_adj = dict(nx.shortest_path_length(nx.from_numpy_matrix(each_edge_weight, create_using=nx.DiGraph()), weight='weight'))

        res = np.zeros_like(self._edges_capacities, dtype=np.float32)
        for dst in range(len(demand_split)):
            dst_demand = np.expand_dims(demand_split[:, dst], 1)
            cost_to_dest = [cost_all_adj[j][dst] for j in range(self._num_nodes)]
            edge_cost = self.__get_edge_cost(cost_to_dest, each_edge_weight)
            softmin_cost_vector = self._softmin(edge_cost)

            cong = self._get_flow_input_vector(softmin_cost_vector, dst_demand, self._eye_masks[dst])
            res += np.reshape(cong, [-1])  # , q_val

        congestion = res / self._edges_capacities
        cost = np.max(congestion)
        return cost, res


# from ecmp_network import *
# from Learning_to_Route.data_generation import tm_generation
# from Learning_to_Route.common.consts import Consts
# import networkx
#
#
# def get_base_graph():
#     # init a triangle if we don't get a network graph
#     g = networkx.Graph()
#     g.add_nodes_from([0, 1, 2, 3, 4, 5])
#     g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (2, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (4, 0, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (3, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (3, 4, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (4, 5, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0}),
#                       (5, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 1, EdgeConsts.TTL_FLOW_STR: 0})])
#
#     return g
#
#
# ecmpNetwork = ECMPNetwork(get_base_graph())
#
# opt = WNumpyOptimizer(ecmpNetwork.get_adjacency, ecmpNetwork.get_capacities())
# tm = tm_generation.one_sample_tm_base(ecmpNetwork, 0.3, Consts.GRAVITY, 0, 0, 0)
# opt.step(np.concatenate([np.ones(16)]), tm)
