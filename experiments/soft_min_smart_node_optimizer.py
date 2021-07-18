"""
Created on 13 May 2021
@author:: Ido Yehezkel
"""

import numpy as np
from common.RL_Env.rl_env_consts import HistoryConsts
from common.static_routing.oblivious_routing import calculate_congestion_per_matrices, oblivious_routing
from common.RL_Env.optimizer_abstract import *
from common.utils import extract_flows, extract_lp_values


class SoftMinSmartNodesOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, testing=False):
        super(SoftMinSmartNodesOptimizer, self).__init__(net, testing)
        self._total_load_per_link = None

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        rl_max_congestion, rl_most_congested_link, rl_total_congestion, \
        rl_total_congestion_per_link, rl_total_load_per_link = self._get_cost_given_weights(weights_vector, traffic_matrix)

        return rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, rl_total_load_per_link

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

    @staticmethod
    def __validate_flow(net_direct: NetworkClass, tm, flows_vars_src2dest_per_edge, all_splitting_ratios):
        splitting_ratios, smart_nodes_spr = all_splitting_ratios
        flows = extract_flows(tm)
        smart_nodes = net_direct.get_smart_nodes

        for src, dst in flows:
            # Flow conservation at the dst
            __flow_from_dst = sum(flows_vars_src2dest_per_edge[src, dst, dst, v] for _, v in net_direct.out_edges_by_node(dst))
            __flow_to_dst = sum(flows_vars_src2dest_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
            error_bound(__flow_to_dst, tm[src, dst])
            error_bound(__flow_from_dst)

            for u in net_direct.nodes:
                if u == dst:
                    continue
                # Flow conservation at src / transit node
                __flow_from_u = sum(flows_vars_src2dest_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                __flow_to_u = sum(flows_vars_src2dest_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                if u == src:
                    error_bound(__flow_from_u, __flow_to_u + tm[src, dst])
                else:
                    error_bound(__flow_from_u, __flow_to_u)

                for _u, v in net_direct.out_edges_by_node(u):
                    assert u == _u
                    del _u
                    u_v_idx = net_direct.get_edge2id(u, v)

                    spr = splitting_ratios[dst, u_v_idx]  # default assignments
                    if u in smart_nodes:
                        src_dst_spr = smart_nodes_spr[src, dst, u_v_idx]
                        spr = src_dst_spr if not np.isnan(src_dst_spr) else spr
                    error_bound(__flow_from_u * spr, flows_vars_src2dest_per_edge[src, dst, u, v])

    def calculating_destination_based_spr(self, weights_vector):
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
            cost_adj = [cost_adj[i] for i in net_direct.nodes]
            edge_cost = self.__get_edge_cost(cost_adj, one_hop_cost)
            q_val = self._soft_min(edge_cost)
            splitting_ratios[node_dst] = q_val
        return splitting_ratios

    def _build_src_dst_hybrid_spr(self, weights_vector, flows):
        destination_based_spr = self.calculating_destination_based_spr(weights_vector)
        net_direct = self._network
        smart_nodes = net_direct.get_smart_nodes
        smart_nodes_spr = net_direct.get_smart_nodes_spr
        src_dst_routing_hybrid_spr = dict()
        for src, dst in flows:
            _src_dst_routing_hybrid_spr = np.copy(destination_based_spr[dst])

            for u, v in net_direct.edges:
                u_v_idx = net_direct.get_edge2id(u, v)
                if u in smart_nodes:
                    _src_dst_routing_hybrid_spr[u_v_idx] = smart_nodes_spr[src, dst, u_v_idx]

            src_dst_routing_hybrid_spr[src, dst] = _src_dst_routing_hybrid_spr

        return src_dst_routing_hybrid_spr

    def _get_cost_given_weights(self, weights_vector, tm):
        flows = extract_flows(tm)
        source_routing_hybrid_spr = self._build_src_dst_hybrid_spr(weights_vector, flows)

        rl_max_congestion, rl_most_congested_link, rl_total_congestion, \
        rl_total_congestion_per_link, rl_total_load_per_link = self._calculating_traffic_distribution(source_routing_hybrid_spr, tm)

        if self._testing:
            logger.info("RL most congested link: {}".format(rl_most_congested_link))
            logger.info("RL MLU: {}".format(rl_max_congestion))

        return rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, rl_total_load_per_link

    def _simulate_flow(self, index, source_routing_hybrid_spr, src, dst, demand):
        net_direct = self._network
        total_load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)
        source_routing_hybrid_spr = source_routing_hybrid_spr * self._outgoing_edges
        online_traffic_vec = np.zeros(shape=(net_direct.get_num_nodes, 1), dtype=np.float64)
        online_traffic_vec[src] = demand
        total_traffic_vec = online_traffic_vec
        completed_demand = 0
        for _ in range(Consts.MAX_ITER):
            if completed_demand / demand >= Consts.PREC_DEMAND:
                break
            online_traffic_by_edge = online_traffic_vec * source_routing_hybrid_spr
            total_load_per_link += np.sum(online_traffic_by_edge, axis=0)
            n2n_online_traffic = online_traffic_by_edge @ self._ingoing_edges.T
            online_traffic_vec = np.reshape(np.sum(n2n_online_traffic, axis=0), newshape=(net_direct.get_num_nodes, 1))
            total_traffic_vec += online_traffic_vec
            completed_demand += online_traffic_vec[dst]
            online_traffic_vec[dst] = 0

        if logger.level == logging.DEBUG:
            for u in net_direct.nodes:
                input_traffic = 0
                if u == src:
                    input_traffic += demand
                for v, _ in net_direct.in_edges_by_node(u):
                    if v == dst:
                        continue
                    v_u_idx = net_direct.get_edge2id(v, u)
                    input_traffic += float(total_traffic_vec[v]) * source_routing_hybrid_spr[v][v_u_idx]
                if error_bound(np.abs(input_traffic / total_traffic_vec[u]), 1e-2):
                    logger.debug("mismatch in simulation flow ({},{}) in node {}".format(src, dst, u))
        self._total_load_per_link[index] = total_load_per_link

    def _calculating_traffic_distribution(self, source_routing_hybrid_spr, tm):
        net_direct = self._network
        flows = extract_flows(tm)
        self._total_load_per_link = np.zeros(shape=(len(flows), net_direct.get_num_edges), dtype=np.float64)
        for index, (src, dst) in enumerate(flows):
            self._simulate_flow(index, source_routing_hybrid_spr[src, dst], src, dst, tm[src, dst])

        total_load_per_link = np.sum(self._total_load_per_link, axis=0)
        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)
        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link
