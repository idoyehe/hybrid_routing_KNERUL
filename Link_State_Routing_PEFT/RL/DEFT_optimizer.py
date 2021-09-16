import numpy as np

from common.RL_Env.optimizer_abstract import *
from math import fsum
import gurobipy as gb
from gurobipy import GRB


class DEFTOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, testing=False):
        super(DEFTOptimizer, self).__init__(net, testing)

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        dst_splitting_ratios = self.calculating_destination_based_spr(weights_vector)
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._calculating_traffic_distribution(dst_splitting_ratios, traffic_matrix)

        return max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def _calculating_exponent_distance_gap(self, weights_vector):
        net_direct = self._network

        assert len(weights_vector) == net_direct.get_num_edges

        reduced_directed_graph = self._build_reduced_weighted_graph(weights_vector)

        distance_gap_by_dest_s_t = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        for t in net_direct.nodes:
            shortest_paths_to_dest = nx.shortest_path_length(G=reduced_directed_graph, target=t, weight=EdgeConsts.WEIGHT_STR)
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                distance_gap_by_dest_s_t[t][u][v] = weights_vector[edge_index] + shortest_paths_to_dest[v] - shortest_paths_to_dest[u]
                assert distance_gap_by_dest_s_t[t][u][v] >= 0

        exp_h_by_dest_s_t = np.exp(-1 * distance_gap_by_dest_s_t)

        return exp_h_by_dest_s_t

    def _calculating_equivalent_number(self, weights_vector, exp_h_by_dest_s_t):
        net_direct = self._network
        equivalent_number_pd = np.empty(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        equivalent_number_pd.fill(np.nan)
        reduced_directed_graph = self._build_reduced_weighted_graph(weights_vector)

        for t in net_direct.nodes:
            shortest_paths_to_dest = nx.shortest_path_length(G=reduced_directed_graph, target=t, weight=EdgeConsts.WEIGHT_STR)
            t_DAG = nx.DiGraph()
            for u, v in net_direct.edges:
                if shortest_paths_to_dest[u] > shortest_paths_to_dest[v]:
                    t_DAG.add_edge(u, v)
            assert nx.is_directed_acyclic_graph(t_DAG)
            reversed_top_sort = list(nx.topological_sort(t_DAG))
            reversed_top_sort.reverse()
            for u in reversed_top_sort:
                if u == t:
                    equivalent_number_pd[t][u] = 1.0
                else:
                    equivalent_number_pd[t][u] = 0
                    for _, v in t_DAG.out_edges(u):
                        assert not np.isnan(equivalent_number_pd[t][v])
                        equivalent_number_pd[t][u] += exp_h_by_dest_s_t[t][u][v] * equivalent_number_pd[t][v]

        return equivalent_number_pd

    def calculating_destination_based_spr(self, weights_vector):
        net_direct = self._network
        exp_h_by_dest_s_t = self._calculating_exponent_distance_gap(weights_vector)
        equivalent_number_by_dst_by_u = self._calculating_equivalent_number(weights_vector, exp_h_by_dest_s_t)

        reduced_directed_graph = self._build_reduced_weighted_graph(weights_vector)

        gamma_pd_by_dst_by_u_v = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        sum_gamma_pd_by_dst_by_u = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)

        for t in net_direct.nodes:
            shortest_paths_to_dest = nx.shortest_path_length(G=reduced_directed_graph, target=t, weight=EdgeConsts.WEIGHT_STR)
            for u, v in net_direct.edges:
                if shortest_paths_to_dest[u] > shortest_paths_to_dest[v]:
                    gamma_pd_by_dst_by_u_v[t][u][v] = equivalent_number_by_dst_by_u[t, v] * exp_h_by_dest_s_t[t][u][v]
                    sum_gamma_pd_by_dst_by_u[t][u] += gamma_pd_by_dst_by_u_v[t][u][v]

        splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                if u == t:
                    continue
                splitting_ratios[t, u, v] = gamma_pd_by_dst_by_u_v[t, u, v] / sum_gamma_pd_by_dst_by_u[t, u]

        for t in net_direct.nodes:
            for u in net_direct.nodes:
                assert error_bound(int(u != t), sum(splitting_ratios[t][u, :]))

        return splitting_ratios
