from common.logger import *
from common.network_class import NetworkClass, nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from common.utils import error_bound


class Optimizer_Abstract(object):
    def __init__(self, net: NetworkClass, testing=False):
        """
        constructor
        @param graph_adjacency_matrix: the graph adjacency matrix
        @param edges_capacities: all edges capacities
        @param max_iterations: number of max iterations
        """
        self._network = net
        self._graph_adjacency_matrix = self._network.get_adjacency
        self._num_nodes = self._network.get_num_nodes
        self._num_edges = self._network.get_num_edges
        self._initialize()
        self._testing = testing

    def _initialize(self):
        logger.debug("Building ingoing and outgoing edges map")
        self._ingoing_edges, self._outgoing_edges, self._edges_capacities = self._network.build_edges_map()

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        pass

    def _calculating_traffic_distribution(self, splitting_ratios, tm):
        net_direct = self._network

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
                    net_direct.in_edges_by_node(s)) + tm[s, t]  # all incoming with originated from s to t

                _outgoing_flow_from_s_destined_t = sum(
                    flows_vars_per_per_dest_per_edge[(t,) + arch] for arch in
                    net_direct.out_edges_by_node(s))
                lp_problem.addConstr(_collected_flow_in_s_destined_t == _outgoing_flow_from_s_destined_t)

                for out_arch in net_direct.out_edges_by_node(s):
                    edge_index = net_direct.get_edge2id(*out_arch)
                    lp_problem.addConstr(flows_vars_per_per_dest_per_edge[(t,) + out_arch] ==
                                         _collected_flow_in_s_destined_t * splitting_ratios[t, edge_index])

        lp_problem.update()

        try:
            logger.debug("LP Submit to Solve {}".format(lp_problem.ModelName))
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

        self.__validate_flow(net_direct, tm, flows_vars_per_per_dest_per_edge, splitting_ratios)

        flows_vars_per_edge_dict = dict()
        total_load_per_link = np.zeros((net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            flows_vars_per_edge_dict[(u, v)] = sum(
                flows_vars_per_per_dest_per_edge[(t, u, v)] for t in net_direct.nodes)
            edge_index = net_direct.get_edge2id(u, v)
            total_load_per_link[edge_index] = flows_vars_per_edge_dict[(u, v)]

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)

        return total_congestion, max_congestion, total_congestion_per_link, most_congested_link

    @staticmethod
    def __validate_flow(net_direct, tm, flows_vars_per_per_dest_per_edge, splitting_ratios):
        for dst in net_direct.nodes:
            for src in net_direct.nodes:
                if src == dst:
                    total_demand_dst = sum(tm[:, dst])
                    assert error_bound(total_demand_dst, sum(
                        flows_vars_per_per_dest_per_edge[dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst)))
                else:
                    _collected_flow_in_s_destined_t = sum(
                        flows_vars_per_per_dest_per_edge[dst, u, src] for u, _ in net_direct.in_edges_by_node(src)) + \
                                                      tm[src, dst]

                    _outgoing_flow_from_s_destined_t = sum(
                        flows_vars_per_per_dest_per_edge[dst, src, u] for _, u in net_direct.out_edges_by_node(src))
                    assert error_bound(_collected_flow_in_s_destined_t, _outgoing_flow_from_s_destined_t)

                    for _, v in net_direct.out_edges_by_node(src):
                        edge_index = net_direct.get_edge2id(src, v)
                        assert error_bound(flows_vars_per_per_dest_per_edge[dst, src, v],
                                           _collected_flow_in_s_destined_t * splitting_ratios[dst, edge_index])
