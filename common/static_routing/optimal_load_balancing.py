import math

from common.consts import EdgeConsts, Consts
from common.utils import error_bound, extract_lp_values
from common.network_class import NetworkClass
from collections import defaultdict
from common.logger import *
import numpy as np
import gurobipy as gb
from gurobipy import GRB


def __validate_solution(net_direct, destinations, tm, flows_dests_per_edge):
    for dst in destinations:
        from_dst = sum(flows_dests_per_edge[dst, dst, v] for _, v in net_direct.out_edges_by_node(dst))
        assert error_bound(from_dst)
        to_dst = sum(flows_dests_per_edge[dst, v, dst] for v, _ in net_direct.in_edges_by_node(dst))
        assert error_bound(to_dst, np.sum(tm, axis=0)[dst])

    for src in net_direct.nodes:
        for dst in destinations:
            if src == dst:
                continue
            from_src = sum(flows_dests_per_edge[dst, src, v] for _, v in net_direct.out_edges_by_node(src))
            to_src = sum(flows_dests_per_edge[dst, v, src] for v, _ in net_direct.in_edges_by_node(src))
            assert error_bound(from_src - to_src, tm[src, dst])


def optimal_load_balancing_LP_solver(net: NetworkClass, traffic_matrix, return_spr=False):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, Consts.OUTPUT_FLAG)
    gb_env.setParam(GRB.Param.NumericFocus, Consts.NUMERIC_FOCUS)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.start()
    return_tuple = aux_optimal_load_balancing_LP_solver(net, traffic_matrix, gb_env, return_spr)
    while True:
        try:
            return_tuple = aux_optimal_load_balancing_LP_solver(net, traffic_matrix, gb_env, return_spr, opt_ratio - 0.001)
            print("****** Gurobi Failure ******")
        except Exception as e:
            return return_tuple


def aux_optimal_load_balancing_LP_solver(net: NetworkClass, traffic_matrix, gurobi_env, return_spr=False, opt_ratio_value=None):
    net_direct = net
    opt_lp_problem = gb.Model(name="LP problem for optimal load balancing, given network and TM", env=gurobi_env)

    destinations = np.where(np.sum(traffic_matrix, axis=0) > 0)[0].tolist()

    vars_flows_dests_per_edge = opt_lp_problem.addVars(destinations, net_direct.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)
    opt_lp_problem.update()
    if opt_ratio_value is None:
        epsilon = 10 ** -np.ceil(np.log10(np.sum(traffic_matrix)))
        congestion_objective = opt_lp_problem.addVar(lb=0.0, name="opt_ratio", vtype=GRB.CONTINUOUS)
        opt_lp_problem.setObjective(congestion_objective + epsilon * vars_flows_dests_per_edge.sum(), sense=GRB.MINIMIZE)

    else:
        congestion_objective = opt_ratio_value
        opt_lp_problem.setObjective(vars_flows_dests_per_edge.sum(), sense=GRB.MINIMIZE)

    opt_lp_problem.update()

    for u, v in net_direct.edges:
        _edge_capacity = net_direct.get_edge_key((u, v), key=EdgeConsts.CAPACITY_STR)
        _edge_load = sum(vars_flows_dests_per_edge[dst, u, v] for dst in destinations)
        opt_lp_problem.addLConstr(_edge_load, GRB.LESS_EQUAL, _edge_capacity * congestion_objective)

    opt_lp_problem.update()

    for node in net_direct.nodes:
        for dst in destinations:
            if node != dst:
                _flow_from_node = sum(vars_flows_dests_per_edge[dst, node, v] for _, v in net_direct.out_edges_by_node(node))
                _flow_to_node = sum(vars_flows_dests_per_edge[dst, u, node] for u, _ in net_direct.in_edges_by_node(node))
                opt_lp_problem.update()
                opt_lp_problem.addLConstr(_flow_from_node - _flow_to_node, GRB.EQUAL, traffic_matrix[node, dst])

    try:
        logger.info("LP Submit to Solve {}".format(opt_lp_problem.ModelName))
        opt_lp_problem.optimize()
        assert opt_lp_problem.Status == GRB.OPTIMAL

    except gb.GurobiError as e:
        raise Exception("Optimize failed due to non-convexity")

    if opt_ratio_value is None:
        opt_ratio_value = congestion_objective.x

    if logger.level == logging.DEBUG:
        opt_lp_problem.printStats()
        opt_lp_problem.printQuality()

    flows_dst_per_edge = extract_lp_values(vars_flows_dests_per_edge)
    del vars_flows_dests_per_edge
    opt_lp_problem.close()

    __validate_solution(net_direct, destinations, traffic_matrix, flows_dst_per_edge)

    necessary_capacity = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
    max_congested_link = 0
    for u, v in net_direct.edges:
        edge_capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
        necessary_capacity[u, v] = sum(flows_dst_per_edge[dst, u, v] for dst in destinations)
        max_congested_link = max(max_congested_link, necessary_capacity[u, v] / edge_capacity)

    assert error_bound(max_congested_link, opt_ratio_value)
    max_congested_link = np.round(max_congested_link, Consts.ROUND)

    if return_spr:
        dst_splitting_ratio = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float)
        for dst in net_direct.nodes:
            for u in net_direct.nodes:
                if u == dst:
                    continue
                flow_out_u = sum(flows_dst_per_edge[(dst, u, v)] for _, v in net_direct.out_edges_by_node(u))
                if flow_out_u > 0.0:
                    for _, v in net_direct.out_edges_by_node(u):
                        dst_splitting_ratio[dst, u, v] = flows_dst_per_edge[(dst, u, v)] / flow_out_u
                    assert error_bound(sum(dst_splitting_ratio[dst][u]), 1.0)

        return max_congested_link, necessary_capacity, dst_splitting_ratio

    return max_congested_link, necessary_capacity
