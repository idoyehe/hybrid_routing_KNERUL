from common.RL_Envs.optimizer_abstract import *
from math import fsum
import gurobipy as gb
from gurobipy import GRB


class PEFTOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, testing=False):
        super(PEFTOptimizer, self).__init__(net, testing)

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

        h_gap_by_dest_u_v = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64) + np.inf
        for node_dst in net_direct.nodes:
            shortest_paths_to_dest = nx.shortest_path_length(G=reduced_directed_graph, target=node_dst, weight=EdgeConsts.WEIGHT_STR)
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                h_gap_by_dest_u_v[node_dst][u, v] = weights_vector[edge_index] + shortest_paths_to_dest[v] - shortest_paths_to_dest[u]
                assert h_gap_by_dest_u_v[node_dst][u, v] >= 0

        exp_h_by_dest_s_t = np.exp(-1 * h_gap_by_dest_u_v)

        return exp_h_by_dest_s_t

    def _calculating_equivalent_number(self, exp_h_by_dest_s_t):
        net_direct = self._network

        gb_env = gb.Env(empty=True)
        gb_env.setParam(GRB.Param.OutputFlag, 0)
        gb_env.setParam(GRB.Param.NumericFocus, 3)
        gb_env.start()

        lp_problem = gb.Model(name="LP problem for gamma t U, given network and exponential penalty values", env=gb_env)

        gammas_by_dest_by_u_vars = lp_problem.addVars(net_direct.get_num_nodes, net_direct.get_num_nodes, name="gamma", vtype=GRB.CONTINUOUS)

        lp_problem.update()

        for t in net_direct.nodes:
            for u in net_direct.nodes:
                if t == u:
                    lp_problem.addConstr(gammas_by_dest_by_u_vars[(t, t)] == 1.0)
                else:
                    sigma = sum(exp_h_by_dest_s_t[t, u, v] * gammas_by_dest_by_u_vars[(t, v)] for _, v in net_direct.out_edges_by_node(u))
                    lp_problem.addConstr(gammas_by_dest_by_u_vars[(t, u)] == sigma)

        lp_problem.update()

        try:
            logger.debug("LP Submit to Solve {}".format(lp_problem.ModelName))
            lp_problem.optimize()
            assert lp_problem.Status == GRB.OPTIMAL

        except AssertionError as e:
            logger_level = logger.level
            logger.setLevel(logging.DEBUG)
            if lp_problem.Status == GRB.UNBOUNDED:
                logger.debug('The model cannot be solved because it is unbounded')
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

            if lp_problem.Status != GRB.INF_OR_UNBD and lp_problem.Status != GRB.INFEASIBLE:
                logger.debug('Optimization was stopped with status {}'.format(lp_problem.Status))
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

            orignumvars = lp_problem.NumVars
            lp_problem.feasRelaxS(0, False, False, True)
            lp_problem.optimize()

            if lp_problem.Status != GRB.OPTIMAL:
                logger.info('Model is infeasible {}'.format(lp_problem.Status))
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

            assert logger.level == logging.DEBUG
            logger.debug('\nSlack values:')
            slacks = lp_problem.getVars()[orignumvars:]
            for sv in slacks:
                if sv.X > 1e-6:
                    logger.debug('{} = {}'.format(sv.VarName, sv.X))

            logger.setLevel(logger_level)

        except gb.GurobiError as e:
            raise Exception("****Optimize failed****\nException is:\n{}".format(e))

        if logger.level == logging.DEBUG:
            lp_problem.printStats()
            lp_problem.printQuality()

        gammas_by_dest_by_u_vars = dict(gammas_by_dest_by_u_vars)

        gammas_by_dest_by_u = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        for key in gammas_by_dest_by_u_vars.keys():
            gammas_by_dest_by_u[key] = gammas_by_dest_by_u_vars[key].x

        return gammas_by_dest_by_u

    def calculating_destination_based_spr(self, weights_vector):
        net_direct = self._network
        exp_h_by_dest_u_v = self._calculating_exponent_distance_gap(weights_vector)
        equiv_num_by_dest_by_u = self._calculating_equivalent_number(exp_h_by_dest_u_v)

        gamma_px_by_dest_by_u_v = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                gamma_px_by_dest_by_u_v[t, edge_index] = equiv_num_by_dest_by_u[t, v] * exp_h_by_dest_u_v[t, u, v]

        sum_gamma_px_by_dest_by_u = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        for t in net_direct.nodes:
            for u in net_direct.nodes:
                sum_gamma_px_by_dest_by_u[t, u] = fsum(gamma_px_by_dest_by_u_v[t, net_direct.get_edge2id(u, v)]
                                                       for _, v in net_direct.out_edges_by_node(u))

        splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                splitting_ratios[t, u, v] = gamma_px_by_dest_by_u_v[t, edge_index] / sum_gamma_px_by_dest_by_u[t, u]
            splitting_ratios[t, t, :] = 0.0

        for t in net_direct.nodes:
            for u in net_direct.nodes:
                assert error_bound(int(t != u), sum(splitting_ratios[t, u, v] for _, v in net_direct.out_edges_by_node(u)))

        return splitting_ratios
