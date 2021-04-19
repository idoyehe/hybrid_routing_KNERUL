from common.network_class import NetworkClass, nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from common.logger import *
from common.RL_Env.optimizer_abstract import *
from math import fsum


class PEFTOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, oblivious_routing_per_edge, testing=False):
        super(PEFTOptimizer, self).__init__(net, testing)
        self._oblivious_routing_per_edge = oblivious_routing_per_edge

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        splitting_ratios = self._calculating_splitting_ratios(weights_vector)
        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
            self._calculating_traffic_distribution(splitting_ratios, traffic_matrix)

        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    def _calculating_exponent_distance_gap(self, weights_vector):
        net_direct = self._network

        assert len(weights_vector) == net_direct.get_num_edges

        reduced_directed_graph = nx.DiGraph()
        for edge_index, cost in enumerate(weights_vector):
            u, v = net_direct.get_id2edge(edge_index)
            reduced_directed_graph.add_edge(u, v, cost=cost)

        assert reduced_directed_graph.number_of_edges() == net_direct.get_num_edges

        distance_gap_by_dest_s_t = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)
        for node_dst in net_direct.nodes:
            shortest_paths_to_dest = nx.shortest_path_length(G=reduced_directed_graph, target=node_dst, weight='cost')
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                distance_gap_by_dest_s_t[node_dst][edge_index] = \
                    weights_vector[edge_index] + shortest_paths_to_dest[v] - shortest_paths_to_dest[u]
                assert distance_gap_by_dest_s_t[node_dst][edge_index] >= 0

        exp_h_by_dest_s_t = np.exp(-1 * distance_gap_by_dest_s_t)

        return exp_h_by_dest_s_t

    def _calculating_equivalent_number(self, exp_h_by_dest_s_t):
        net_direct = self._network

        gb_env = gb.Env(empty=True)
        gb_env.setParam(GRB.Param.OutputFlag, 0)
        gb_env.setParam(GRB.Param.NumericFocus, 3)
        gb_env.start()

        lp_problem = gb.Model(name="LP problem for gamma t U, given network and exponential penalty values", env=gb_env)

        gammas_by_dest_by_u_vars = lp_problem.addVars(net_direct.get_num_nodes, net_direct.get_num_nodes, name="gamma",
                                                      vtype=GRB.CONTINUOUS)

        lp_problem.update()

        for t in net_direct.nodes:
            for u in net_direct.nodes:
                if t == u:
                    lp_problem.addConstr(gammas_by_dest_by_u_vars[(t, t)] == 1.0)
                else:
                    sigma = sum(
                        exp_h_by_dest_s_t[(t, net_direct.get_edge2id(u, v))] * gammas_by_dest_by_u_vars[(t, v)]
                        for _, v in net_direct.out_edges_by_node(u))
                    lp_problem.addConstr(gammas_by_dest_by_u_vars[(t, u)] == sigma)

        lp_problem.update()

        try:
            logger.info("LP Submit to Solve {}".format(lp_problem.ModelName))
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

    def _calculating_splitting_ratios(self, weights_vector):
        net_direct = self._network
        exp_h_by_dest_s_t = self._calculating_exponent_distance_gap(weights_vector)
        gammas_by_dest_by_u = self._calculating_equivalent_number(exp_h_by_dest_s_t)

        gamma_px_by_dest_by_u_v = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges),
                                           dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                gamma_px_by_dest_by_u_v[t, edge_index] = gammas_by_dest_by_u[t, v] * exp_h_by_dest_s_t[t, edge_index]

        sum_gamma_px_by_dest_by_u = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        for t in net_direct.nodes:
            for u in net_direct.nodes:
                sum_gamma_px_by_dest_by_u[t, u] = fsum(gamma_px_by_dest_by_u_v[t, net_direct.get_edge2id(u, v)]
                                                       for _, v in net_direct.out_edges_by_node(u))

        splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id(u, v)
                splitting_ratios[t, edge_index] = gamma_px_by_dest_by_u_v[t, edge_index] / sum_gamma_px_by_dest_by_u[
                    t, u]

        for t in net_direct.nodes:
            for u in net_direct.nodes:
                assert error_bound(1.0, sum(
                    splitting_ratios[t, net_direct.get_edge2id(u, v)] for _, v in net_direct.out_edges_by_node(u)))

        return splitting_ratios


if __name__ == "__main__":
    from common.topologies import BASIC_TOPOLOGIES
    from common.static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver

    ecmpNetwork = NetworkClass(BASIC_TOPOLOGIES["TRIANGLE"])
    tm = np.array([[0, 10, 0], [0, 0, 0], [0, 0, 0]])

    opt = PEFTOptimizer(ecmpNetwork, None)
    opt_congestion, opt_routing_scheme = optimal_load_balancing_LP_solver(net=ecmpNetwork, traffic_matrix=tm)
    print("Optimal Congestion: {}".format(opt_congestion))
    max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
        opt.step([5, 2.5, 100, 100, 100, 2.5], tm, opt_congestion)
    print("Optimizer Congestion: {}".format(max_congestion))
    print("Optimizer Most Congested Link: {}".format(ecmpNetwork.get_id2edge(most_congested_link)))
    print("Congestion Ratio :{}".format(max_congestion / opt_congestion))
