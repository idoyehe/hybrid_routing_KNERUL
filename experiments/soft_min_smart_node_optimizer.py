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

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        rl_max_congestion, rl_most_congested_link, rl_total_congestion, \
        rl_total_congestion_per_link, rl_total_load_per_link = self._get_cost_given_weights(weights_vector, traffic_matrix, optimal_value)

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

    def calculating_splitting_ratios(self, weights_vector):
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

    def _get_cost_given_weights(self, weights_vector, tm, optimal_value):
        splitting_ratios = self.calculating_splitting_ratios(weights_vector)

        rl_max_congestion, rl_most_congested_link, rl_total_congestion, \
        rl_total_congestion_per_link, rl_total_load_per_link = self._calculating_traffic_distribution(splitting_ratios, tm)

        if self._testing:
            logger.info("RL most congested link: {}".format(rl_most_congested_link))
            logger.info("RL cost value: {}".format(rl_max_congestion / optimal_value))

        return rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, rl_total_load_per_link

    def _calculating_traffic_distribution(self, splitting_ratios, tm):
        net_direct = self._network
        smart_nodes = net_direct.get_smart_nodes
        smart_nodes_spr = net_direct.get_smart_nodes_spr

        self._initialize_gurobi_env()
        self.gb_env.start()

        mcf_problem = gb.Model(name="LP problem for flows, given network, traffic matrix and splitting_ratios", env=self.gb_env)
        flows = extract_flows(tm)

        flows_vars_src2dest_per_edge = mcf_problem.addVars(flows, net_direct.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)

        for src, dst in flows:
            # Flow conservation at the dst
            __flow_from_dst = sum(flows_vars_src2dest_per_edge[src, dst, dst, v] for _, v in net_direct.out_edges_by_node(dst))
            __flow_to_dst = sum(flows_vars_src2dest_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
            mcf_problem.addConstr(__flow_to_dst == tm[src, dst])
            mcf_problem.addConstr(__flow_from_dst == 0.0)

            for u in net_direct.nodes:
                if u == dst:
                    continue
                # Flow conservation at src / transit node
                __flow_from_u = sum(flows_vars_src2dest_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                __flow_to_u = sum(flows_vars_src2dest_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                if u == src:
                    mcf_problem.addConstr(__flow_from_u == __flow_to_u + tm[src, dst])
                else:
                    mcf_problem.addConstr(__flow_from_u == __flow_to_u)

                for _u, v in net_direct.out_edges_by_node(u):
                    assert u == _u
                    del _u
                    u_v_idx = net_direct.get_edge2id(u, v)

                    spr = splitting_ratios[dst, u_v_idx]  # default assignments
                    if u in smart_nodes:
                        src_dst_spr = smart_nodes_spr[src, dst, u_v_idx]
                        spr = src_dst_spr if not np.isnan(src_dst_spr) else spr

                    mcf_problem.addConstr(__flow_from_u * spr == flows_vars_src2dest_per_edge[src, dst, u, v])

            mcf_problem.update()

        try:
            logger.debug("LP Submit to Solve {}".format(mcf_problem.ModelName))
            mcf_problem.optimize()
            assert mcf_problem.Status == GRB.OPTIMAL
        except AssertionError as e:
            logger_level = logger.level
            logger.setLevel(logging.DEBUG)
            if mcf_problem.Status == GRB.UNBOUNDED:
                logger.debug('The model cannot be solved because it is unbounded')
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(mcf_problem.Status))

            if mcf_problem.Status != GRB.INF_OR_UNBD and mcf_problem.Status != GRB.INFEASIBLE:
                logger.debug('Optimization was stopped with status {}'.format(mcf_problem.Status))
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(mcf_problem.Status))

            orignumvars = mcf_problem.NumVars
            mcf_problem.feasRelaxS(0, False, False, True)
            mcf_problem.optimize()

            if mcf_problem.Status != GRB.OPTIMAL:
                tm_file_name = "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\tm.npy"
                tm_file = open(tm_file_name, 'wb')
                np.save(tm_file, tm)
                tm_file.close()

                sr_file_name = "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\sr.npy"
                sr_file = open(sr_file_name, 'wb')
                np.save(sr_file, splitting_ratios)
                sr_file.close()
                logger.info('Model is infeasible {}'.format(mcf_problem.Status))
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(mcf_problem.Status))

            assert logger.level == logging.DEBUG
            slacks = mcf_problem.getVars()[orignumvars:]
            for sv in slacks:
                if sv.X > Consts.FEASIBILITY_TOL:
                    logger.debug('Slack value:{} = {}'.format(sv.VarName, sv.X))

            logger.setLevel(logger_level)

        except gb.GurobiError as e:
            raise Exception("****Optimize failed****\nException is:\n{}".format(e))

        if logger.level == logging.DEBUG:
            mcf_problem.printStats()
            mcf_problem.printQuality()

        flows_vars_src2dest_per_edge = extract_lp_values(flows_vars_src2dest_per_edge)
        mcf_problem.close()
        self.gb_env.close()

        self.__validate_flow(net_direct, tm, flows_vars_src2dest_per_edge, (splitting_ratios, smart_nodes_spr))

        flows_vars_per_edge_dict = dict()
        total_load_per_link = np.zeros((net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            flows_vars_per_edge_dict[(u, v)] = sum(flows_vars_src2dest_per_edge[(src, dst, u, v)] for src, dst in flows)
            edge_index = net_direct.get_edge2id(u, v)
            total_load_per_link[edge_index] = flows_vars_per_edge_dict[(u, v)]

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)

        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link
