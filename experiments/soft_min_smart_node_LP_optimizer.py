"""
Created on 13 May 2021
@author:: Ido Yehezkel
"""
from common.RL_Env.rl_env_consts import HistoryConsts
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

    def _get_cost_given_weights(self, weights_vector, tm, optimal_value):
        flows = extract_flows(tm)
        src_dst_splitting_ratios = self.calculating_src_dst_spr(weights_vector, flows)

        rl_max_congestion, rl_most_congested_link, rl_total_congestion, \
        rl_total_congestion_per_link, rl_total_load_per_link = self._calculating_traffic_distribution(src_dst_splitting_ratios, tm)
        if self._testing:
            logger.info("RL most congested link: {}".format(rl_most_congested_link))
            logger.info("RL MLU: {}".format(rl_max_congestion))
            logger.info("RL MLU Vs. Optimal: {}".format(optimal_value))

        return rl_max_congestion, rl_most_congested_link, rl_total_congestion, rl_total_congestion_per_link, rl_total_load_per_link

    def calculating_destination_based_spr(self, weights_vector):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network
        dst_splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)
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
            dst_splitting_ratios[node_dst] = q_val
        return dst_splitting_ratios

    def calculating_src_dst_spr(self, weights_vector, flows):
        logger.debug("Calculating hop by hop splitting ratios")
        net_direct = self._network
        smart_nodes_spr = net_direct.get_smart_nodes_spr

        src_dst_splitting_ratios = dict()
        dst_splitting_ratios = self.calculating_destination_based_spr(weights_vector)
        for src, dst in flows:
            src_dst_splitting_ratios[(src, dst)] = np.copy(dst_splitting_ratios[dst])
            for s_n in net_direct.get_smart_nodes:
                for _, v in net_direct.out_edges_by_node(s_n):
                    u_v_idx = net_direct.get_edge2id(s_n, v)
                    src_dst_spr = smart_nodes_spr[src, dst, u_v_idx]
                    if not np.isnan(src_dst_spr):
                        src_dst_splitting_ratios[(src, dst)][u_v_idx] = src_dst_spr

        return src_dst_splitting_ratios

    def _calculating_traffic_distribution(self, src_dst_splitting_ratios, tm):
        net_direct = self._network

        self._initialize_gurobi_env()
        self.gb_env.start()

        mcf_problem = gb.Model(name="LP problem for flows, given network, traffic matrix and splitting_ratios", env=self.gb_env)
        flows = extract_flows(tm)

        flows_vars_src2dest_per_node = mcf_problem.addVars(flows, net_direct.nodes, name="f", lb=0.0, vtype=GRB.CONTINUOUS)
        mcf_problem.update()

        for src, dst in flows:
            current_spr = src_dst_splitting_ratios[src, dst]
            mcf_problem.addLConstr(flows_vars_src2dest_per_node[src, dst, dst], GRB.EQUAL, tm[src, dst])  # Flow conservation at the dst
            for node in net_direct.nodes:
                _flow_to_node = sum(
                    flows_vars_src2dest_per_node[src, dst, u] * current_spr[net_direct.get_edge2id(u, v)] if u != dst else 0 for u, v in
                    net_direct.in_edges_by_node(node))
                if node == src:
                    mcf_problem.addLConstr(_flow_to_node + tm[src, dst], GRB.EQUAL, flows_vars_src2dest_per_node[src, dst, node])
                else:
                    mcf_problem.addLConstr(_flow_to_node, GRB.EQUAL, flows_vars_src2dest_per_node[src, dst, node])
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

        flows_src2dest_per_node = extract_lp_values(flows_vars_src2dest_per_node)
        mcf_problem.close()
        self.gb_env.close()

        self.__validate_flow(net_direct, tm, flows_src2dest_per_node, src_dst_splitting_ratios)

        total_load_per_link = np.zeros((net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            edge_index = net_direct.get_edge2id(u, v)
            total_load_per_link[edge_index] = sum(
                flows_src2dest_per_node[(src, dst)][u] * src_dst_splitting_ratios[src, dst][u, v] if u != dst else 0 for src, dst in flows)

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)

        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    @staticmethod
    def __validate_flow(net_direct: NetworkClass, tm, flows_src2dest_per_node, src_dst_splitting_ratios):
        flows = extract_flows(tm)
        for src, dst in flows:
            current_spr = src_dst_splitting_ratios[src, dst]
            assert flows_src2dest_per_node[src, dst, src] >= tm[src, dst]
            assert error_bound(flows_src2dest_per_node[src, dst, dst], tm[src, dst])
            for node in net_direct.nodes:
                _flow_to_node = sum(flows_src2dest_per_node[src, dst, u] * current_spr[net_direct.get_edge2id(u, v)] if u != dst else 0 for u, v in
                                    net_direct.in_edges_by_node(node))
                if node == src:
                    assert error_bound(flows_src2dest_per_node[src, dst, node], _flow_to_node + tm[src, dst])
                else:
                    assert error_bound(flows_src2dest_per_node[src, dst, node], _flow_to_node)
