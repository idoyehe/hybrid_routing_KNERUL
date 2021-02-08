from common.network_class import NetworkClass, nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from common.logger import *
from common.utils import error_bound
from common.optimizer_abstract import Optimizer_Abstract
from math import fsum


class PEFTOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, oblivious_routing_per_edge=None, max_iterations=500, testing=False):
        """
        constructor
        @param graph_adjacency_matrix: the graph adjacency matrix
        @param edges_capacities: all edges capacities
        @param max_iterations: number of max iterations
        """
        self._network = net
        self._graph_adjacency_matrix = self._network.get_adjacency
        self._num_nodes = self._network.get_num_nodes
        self._initialize()

    def _initialize(self):
        logger.debug("Building ingoing and outgoing edges map")
        _, self._ingoing_edges, self._outgoing_edges, self._edges_capacities = self._network.build_edges_map()

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        total_congestion, max_congestion, total_load_per_arch, most_congested_arch = \
            self._calculating_traffic_distribution(weights_vector, traffic_matrix, optimal_value)

        return total_congestion, max_congestion, total_load_per_arch, most_congested_arch

    def _calculating_exponent_distance_gap(self, weights_vector):
        net = self._network.get_g_directed
        net_direct = net.get_g_directed
        del net

        assert len(weights_vector) == net_direct.get_num_edges

        reduced_directed_graph = nx.DiGraph()
        for edge_index, cost in enumerate(weights_vector):
            u, v = net_direct.get_id2edge()[edge_index]
            reduced_directed_graph.add_edge(u, v, cost=cost)

        assert reduced_directed_graph.number_of_edges() == net_direct.get_num_edges

        distance_gap_by_dest_s_t = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)
        for node_dst in net_direct.nodes:
            shortest_paths_to_dest = nx.shortest_path_length(G=reduced_directed_graph, target=node_dst, weight='cost')
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id()[(u, v)]
                distance_gap_by_dest_s_t[node_dst][edge_index] = \
                    weights_vector[edge_index] + shortest_paths_to_dest[v] - shortest_paths_to_dest[u]
                assert distance_gap_by_dest_s_t[node_dst][edge_index] >= 0

        exp_h_by_dest_s_t = np.exp(-1 * distance_gap_by_dest_s_t)

        return exp_h_by_dest_s_t

    def _calculating_equivalent_number(self, exp_h_by_dest_s_t):
        net = self._network.get_g_directed
        net_direct = net.get_g_directed
        del net

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
                        exp_h_by_dest_s_t[(t, net_direct.get_edge2id()[u, v])] * gammas_by_dest_by_u_vars[(t, v)]
                        for _, v in net_direct.out_edges_by_node(u))
                    lp_problem.addConstr(gammas_by_dest_by_u_vars[(t, u)] == sigma)

        lp_problem.update()

        try:
            logger.info("LP Submit to Solve {}".format(lp_problem.ModelName))
            lp_problem.optimize()
            assert lp_problem.Status == GRB.OPTIMAL
        except AssertionError as e:
            raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

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
        net = self._network.get_g_directed
        net_direct = net.get_g_directed
        del net

        exp_h_by_dest_s_t = self._calculating_exponent_distance_gap(weights_vector)
        gammas_by_dest_by_u = self._calculating_equivalent_number(exp_h_by_dest_s_t)

        gamma_px_by_dest_by_u_v = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges),
                                           dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id()[u, v]
                gamma_px_by_dest_by_u_v[t, edge_index] = gammas_by_dest_by_u[t, v] * exp_h_by_dest_s_t[t, edge_index]

        sum_gamma_px_by_dest_by_u = np.zeros((net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
        for t in net_direct.nodes:
            for u in net_direct.nodes:
                sum_gamma_px_by_dest_by_u[t, u] = fsum(gamma_px_by_dest_by_u_v[t, net_direct.get_edge2id()[u, v]]
                                                       for _, v in net_direct.out_edges_by_node(u))

        splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)

        for t in net_direct.nodes:
            for u, v in net_direct.edges:
                edge_index = net_direct.get_edge2id()[u, v]
                splitting_ratios[t, edge_index] = gamma_px_by_dest_by_u_v[t, edge_index] / sum_gamma_px_by_dest_by_u[
                    t, u]

        # for t in net_direct.nodes:
        #     for u in net_direct.nodes:
        #         assert error_bound(1.0, fsum(
        #             splitting_ratios[t, net_direct.get_edge2id()[u, v]] for _, v in net_direct.out_edges_by_node(u)))

        return splitting_ratios

    def _calculating_traffic_distribution(self, weights_vector, tm, optimal_value):
        splitting_ratios = self._calculating_splitting_ratios(weights_vector)
        net = self._network.get_g_directed
        net_direct = net.get_g_directed
        del net

        gb_env = gb.Env(empty=True)
        gb_env.setParam(GRB.Param.OutputFlag, 0)
        gb_env.setParam(GRB.Param.NumericFocus, 3)
        gb_env.start()

        lp_problem = gb.Model(name="LP problem for flows, given network, traffic matrix and splitting_ratios",
                              env=gb_env)
        flows_vars_per_per_dest_per_edge = lp_problem.addVars(net_direct.nodes, net_direct.edges, name="f", lb=0.0,
                                                              vtype=GRB.CONTINUOUS)

        for s in net_direct.nodes:
            for t in net_direct.nodes:
                if s == t:
                    lp_problem.addConstrs(
                        (flows_vars_per_per_dest_per_edge[(t,) + arch] == 0 for arch in
                         net_direct.out_edges_by_node(t)))
                    _collected_flow_in_t_destined_t = sum(
                        flows_vars_per_per_dest_per_edge[(t,) + arch] for arch in
                        net_direct.in_edges_by_node(t))
                    lp_problem.addConstr(_collected_flow_in_t_destined_t == sum(tm[:, t]))
                    continue

                _collected_flow_in_s_destined_t = sum(
                    flows_vars_per_per_dest_per_edge[(t,) + arch] for arch in
                    net_direct.in_edges_by_node(s)) + tm[s, t]

                _outgoing_flow_from_s_destined_t = sum(
                    flows_vars_per_per_dest_per_edge[(t,) + arch] for arch in
                    net_direct.out_edges_by_node(s))
                lp_problem.addConstr(_collected_flow_in_s_destined_t == _outgoing_flow_from_s_destined_t)

                for out_arch in net_direct.out_edges_by_node(s):
                    edge_index = net_direct.get_edge2id()[out_arch]
                    lp_problem.addConstr(flows_vars_per_per_dest_per_edge[(t,) + out_arch] ==
                                         _collected_flow_in_s_destined_t * splitting_ratios[t, edge_index])

        lp_problem.update()

        try:
            logger.info("LP Submit to Solve {}".format(lp_problem.ModelName))
            lp_problem.optimize()
            assert lp_problem.Status == GRB.OPTIMAL
        except AssertionError as e:
            raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

        except gb.GurobiError as e:
            raise Exception("****Optimize failed****\nException is:\n{}".format(e))

        if logger.level == logging.DEBUG:
            lp_problem.printStats()
            lp_problem.printQuality()

        flows_vars_per_per_dest_per_edge = dict(flows_vars_per_per_dest_per_edge)
        for key in flows_vars_per_per_dest_per_edge.keys():
            flows_vars_per_per_dest_per_edge[key] = flows_vars_per_per_dest_per_edge[key].x

        flows_vars_per_edge_dict = dict()
        total_load_per_link = np.zeros((net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            flows_vars_per_edge_dict[(u, v)] = sum(
                flows_vars_per_per_dest_per_edge[(t, u, v)] for t in net_direct.nodes)
            edge_index = net_direct.get_edge2id()[(u, v)]
            total_load_per_link[edge_index] = flows_vars_per_edge_dict[(u, v)]

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)

        return total_congestion, max_congestion, total_congestion_per_link, most_congested_link


if __name__ == "__main__":
    from topologies import BASIC_TOPOLOGIES
    from static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver

    ecmpNetwork = NetworkClass(BASIC_TOPOLOGIES["TRIANGLE"]).get_g_directed
    tm = np.array([[0, 10, 0], [0, 0, 0], [0, 0, 0]])

    opt = PEFTOptimizer(ecmpNetwork, None)
    opt_congestion, opt_routing_scheme = optimal_load_balancing_LP_solver(net=ecmpNetwork, traffic_matrix=tm)
    print("Optimal Congestion: {}".format(opt_congestion))
    total_congestion, max_congestion, total_load_per_arch, most_congested_arch = opt.step(
        [5, 2.5, 100, 100, 100, 2.5], tm, opt_congestion)
    print("Optimizer Congestion: {}".format(max_congestion))
    print("Optimizer Most Congested Link: {}".format(ecmpNetwork.get_id2edge()[most_congested_arch]))
    print("Congestion Ratio :{}".format(max_congestion / opt_congestion))
