from common.consts import EdgeConsts
from common.utils import extract_flows, error_bound
from common.network_class import NetworkClass
from collections import defaultdict
from common.logger import *
import numpy as np
import gurobipy as gb
from gurobipy import GRB

R = 10


def __extract_values(gurobi_vars_dict):
    gurobi_vars_dict = dict(gurobi_vars_dict)

    for key in gurobi_vars_dict.keys():
        gurobi_vars_dict[key] = round(gurobi_vars_dict[key].x, R)
    return gurobi_vars_dict


def __validate_splitting_ratios(net_direct, flows, splitting_ratios_per_src_dst_edge):
    for u in net_direct.nodes():
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            splitting_ratios_per_src_dst_u_list = [splitting_ratios_per_src_dst_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u)]
            if all(spt is None for spt in splitting_ratios_per_src_dst_u_list):
                continue
            elif all(spt >= 0 for spt in splitting_ratios_per_src_dst_u_list):
                assert error_bound(sum(splitting_ratios_per_src_dst_u_list), 1.0)
            else:
                raise Exception("Splitting ratio are not valid")


def __validate_flows(net_direct, tm, flows_per_edge_src_dst, splitting_ratios_per_src_dst_edge):
    current_flows = extract_flows(tm)

    for src, dst in current_flows:
        for u in net_direct.nodes:
            if u == src:
                from_src = sum(flows_per_edge_src_dst[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(from_src, tm[src, dst])
                to_src = sum(flows_per_edge_src_dst[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                assert error_bound(to_src)

                for _, v in net_direct.out_edges_by_node(u):
                    assert error_bound(flows_per_edge_src_dst[src, dst, u, v], from_src * splitting_ratios_per_src_dst_edge[src, dst, u, v])

            elif u == dst:
                from_dst = sum(flows_per_edge_src_dst[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(from_dst)

                to_dst = sum(flows_per_edge_src_dst[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                assert error_bound(to_dst, tm[src, dst])

            else:
                assert u not in (src, dst)
                to_transit_u = sum(flows_per_edge_src_dst[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                from_transit_u = sum(flows_per_edge_src_dst[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(to_transit_u, from_transit_u)
                if from_transit_u > 0:
                    for _, v in net_direct.out_edges_by_node(u):
                        assert error_bound(flows_per_edge_src_dst[src, dst, u, v], from_transit_u * splitting_ratios_per_src_dst_edge[src, dst, u, v])

    for key, value in flows_per_edge_src_dst.items():
        if (key[0], key[1]) not in current_flows:
            assert value == 0.0


def __validate_solution(net_direct, flows, traffic_matrix, splitting_ratios_per_src_dst_edge, flows_src_dst_per_edge):
    __validate_splitting_ratios(net_direct, flows, splitting_ratios_per_src_dst_edge)
    __validate_flows(net_direct, traffic_matrix, flows_src_dst_per_edge, splitting_ratios_per_src_dst_edge)


def optimal_load_balancing_LP_solver(net: NetworkClass, traffic_matrix):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 3)
    gb_env.setParam(GRB.Param.FeasibilityTol, 1e-9)
    gb_env.start()
    opt_ratio, necessary_capacity_dict, splitting_ratios_per_src_dst_edge = aux_optimal_load_balancing_LP_solver(net, traffic_matrix, gb_env)
    while True:
        try:
            opt_ratio, necessary_capacity_dict, splitting_ratios_per_src_dst_edge = \
                aux_optimal_load_balancing_LP_solver(net, traffic_matrix, gb_env, opt_ratio)
            print("****** Gurobi Failure ******")
            opt_ratio -= 0.0001
        except Exception as e:
            return opt_ratio, necessary_capacity_dict, splitting_ratios_per_src_dst_edge


def aux_optimal_load_balancing_LP_solver(net: NetworkClass, traffic_matrix, gurobi_env, opt_ratio_value=None):
    opt_lp_problem = gb.Model(name="LP problem for optimal load balancing, given network and TM", env=gurobi_env)

    flows = extract_flows(traffic_matrix)

    arch_all_vars = defaultdict(list)

    net_direct = net

    vars_flows_src_dst_per_edge = opt_lp_problem.addVars(flows, net_direct.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)
    opt_lp_problem.update()

    for flow in flows:
        src, dst = flow
        assert src != dst
        assert traffic_matrix[flow] > 0
        for _arch in net_direct.edges:
            g_var = vars_flows_src_dst_per_edge[flow + _arch]
            opt_lp_problem.addConstr(g_var <= traffic_matrix[flow])
            arch_all_vars[_arch].append(g_var)

    opt_lp_problem.update()

    opt_lp_problem.setObjectiveN(sum(dict(vars_flows_src_dst_per_edge).values()), 1)

    if opt_ratio_value is None:
        opt_ratio = opt_lp_problem.addVar(lb=0.0, name="opt_ratio", vtype=GRB.CONTINUOUS)

        opt_lp_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        opt_lp_problem.setObjectiveN(opt_ratio, 0, 1)
        opt_lp_problem.update()
    else:
        opt_ratio = opt_ratio_value

    for _arch in net_direct.edges:
        _arch_capacity = net_direct.get_edge_key(_arch, key=EdgeConsts.CAPACITY_STR)
        opt_lp_problem.addConstr(sum(arch_all_vars[_arch]) <= _arch_capacity * opt_ratio)

    opt_lp_problem.update()

    for flow in flows:
        src, dst = flow
        # Flow conservation at the source
        from_its_src = sum(vars_flows_src_dst_per_edge[flow + out_arch] for out_arch in net_direct.out_edges_by_node(src))
        to_its_src = sum(vars_flows_src_dst_per_edge[flow + in_arch] for in_arch in net_direct.in_edges_by_node(src))
        opt_lp_problem.addConstr(from_its_src == traffic_matrix[flow])
        opt_lp_problem.addConstr(to_its_src == 0)

        # Flow conservation at the destination
        from_its_dst = sum(vars_flows_src_dst_per_edge[flow + out_arch] for out_arch in net_direct.out_edges_by_node(dst))
        to_its_dst = sum(vars_flows_src_dst_per_edge[flow + in_arch] for in_arch in net_direct.in_edges_by_node(dst))
        opt_lp_problem.addConstr(to_its_dst == traffic_matrix[flow])
        opt_lp_problem.addConstr(from_its_dst == 0)

        for u in net_direct.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            from_some_u = sum(vars_flows_src_dst_per_edge[flow + out_arch] for out_arch in net_direct.out_edges_by_node(u))
            to_some_u = sum(vars_flows_src_dst_per_edge[flow + in_arch] for in_arch in net_direct.in_edges_by_node(u))
            opt_lp_problem.addConstr(from_some_u == to_some_u)
        opt_lp_problem.update()

    try:
        logger.info("LP Submit to Solve {}".format(opt_lp_problem.ModelName))
        opt_lp_problem.optimize()
        assert opt_lp_problem.Status == GRB.OPTIMAL

    except gb.GurobiError as e:
        raise Exception("Optimize failed due to non-convexity")

    if opt_ratio_value is None:
        opt_ratio = opt_lp_problem.objVal

    if logger.level == logging.DEBUG:
        opt_lp_problem.printStats()
        opt_lp_problem.printQuality()

    flows_src_dst_per_edge = __extract_values(vars_flows_src_dst_per_edge)
    del vars_flows_src_dst_per_edge
    opt_lp_problem.close()

    splitting_ratios_per_src_dst_edge = dict()
    for u in net_direct.nodes:
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            flow_from_u_to_dst = sum(flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            if flow_from_u_to_dst > 0:
                for _, v in net_direct.out_edges_by_node(u):
                    splitting_ratios_per_src_dst_edge[src, dst, u, v] = flows_src_dst_per_edge[src, dst, u, v] / flow_from_u_to_dst
                    assert splitting_ratios_per_src_dst_edge[src, dst, u, v] >= 0

            else:
                for _, v in net_direct.out_edges_by_node(u):
                    splitting_ratios_per_src_dst_edge[src, dst, u, v] = 1 / len(net_direct.out_edges_by_node(u))
                    assert splitting_ratios_per_src_dst_edge[src, dst, u, v] >= 0

    __validate_solution(net_direct, flows, traffic_matrix, splitting_ratios_per_src_dst_edge, flows_src_dst_per_edge)

    necessary_capacity_dict = dict()
    for u, v in net_direct.edges:
        necessary_capacity_dict[u, v] = sum(flows_src_dst_per_edge[src, dst, u, v] for src, dst in flows)

    max_congested_link = 0

    for u, v in net_direct.edges:
        link_capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
        max_congested_link = max(max_congested_link, necessary_capacity_dict[u, v] / link_capacity)

    assert error_bound(max_congested_link, opt_ratio)
    return round(max_congested_link, 4), necessary_capacity_dict, splitting_ratios_per_src_dst_edge
