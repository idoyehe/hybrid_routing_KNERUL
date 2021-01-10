from common.consts import EdgeConsts
from Learning_to_Route.common.utils import extract_flows, error_bound
from common.network_class import NetworkClass, nx
from collections import defaultdict
from common.logger import *
import numpy as np
import gurobipy as gb
from gurobipy import GRB


def __validate_solution(net_directed: NetworkClass, flows: list, traffic_matrix, arch_g_vars_dict):
    assert net_directed.g_is_directed

    for flow in flows:
        src, dst = flow
        for v in net_directed.nodes:
            if v == src:
                from_its_source = 0
                for out_arches_from_src in net_directed.out_edges_by_node(v):
                    from_its_source += arch_g_vars_dict[out_arches_from_src][flow]
                assert error_bound(from_its_source, traffic_matrix[flow])
                to_its_src = 0
                for in_arches_to_src in net_directed.in_edges_by_node(v):
                    to_its_src += arch_g_vars_dict[in_arches_to_src][flow]
                assert error_bound(to_its_src)

            elif v == dst:
                from_its_dst = 0
                for out_arches_from_dst in net_directed.out_edges_by_node(v):
                    from_its_dst += arch_g_vars_dict[out_arches_from_dst][flow]
                assert error_bound(from_its_dst)

                to_its_dst = 0
                for in_arches_to_dst in net_directed.in_edges_by_node(v):
                    to_its_dst += arch_g_vars_dict[in_arches_to_dst][flow]
                assert error_bound(to_its_dst, traffic_matrix[flow])
            else:
                assert v not in flow
                to_some_v = 0
                for in_arches_to_v in net_directed.in_edges_by_node(v):
                    to_some_v += arch_g_vars_dict[in_arches_to_v][flow]
                from_some_v = 0
                for out_arches_from_v in net_directed.out_edges_by_node(v):
                    from_some_v += arch_g_vars_dict[out_arches_from_v][flow]
                assert error_bound(to_some_v, from_some_v)


def optimal_load_balancing_LP_solver(net: NetworkClass, traffic_matrix):
    prev_opt_ratio, prev_link_carries_per_flow = aux_optimal_load_balancing_LP_solver(net, traffic_matrix)
    while True:
        try:
            next_opt_ratio = prev_opt_ratio - 0.001
            prev_opt_ratio, prev_link_carries_per_flow = aux_optimal_load_balancing_LP_solver(net, traffic_matrix,
                                                                                              next_opt_ratio)
            print("****** Gurobi Failure ******")
            prev_opt_ratio = next_opt_ratio
        except Exception as e:
            return prev_opt_ratio, prev_link_carries_per_flow


def aux_optimal_load_balancing_LP_solver(net: NetworkClass, traffic_matrix, opt_ratio_value=None):
    opt_lp_problem = gb.Model(name="LP problem for optimal load balancing, given network and TM")

    flows = extract_flows(traffic_matrix)

    arch_vars_per_flow = defaultdict(dict)
    arch_all_vars = defaultdict(list)

    net_direct = net.get_g_directed
    all_vars_sum = 0

    for flow in flows:
        src, dst = flow
        assert src != dst
        assert traffic_matrix[flow] > 0
        for _arch in net_direct.edges:
            g_var_name = "arch{};flow{}->{};".format(str(_arch), src, dst)
            g_var = opt_lp_problem.addVar(lb=0.0, name=g_var_name, vtype=GRB.CONTINUOUS)
            all_vars_sum += g_var

            arch_vars_per_flow[_arch][flow] = g_var
            arch_all_vars[_arch].append(g_var)

    opt_lp_problem.update()

    opt_lp_problem.setObjectiveN(all_vars_sum, 1)

    if opt_ratio_value is None:
        opt_ratio = opt_lp_problem.addVar(lb=0.0, name="opt_ratio", vtype=GRB.CONTINUOUS)
        opt_lp_problem.setParam(GRB.Param.OutputFlag, 0)
        # opt_lp_problem.setParam(GRB.Param.BarConvTol, 0.01)
        # opt_lp_problem.setParam(GRB.Param.Method, 2)
        # opt_lp_problem.setParam(GRB.Param.Crossover, 0)

        opt_lp_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        opt_lp_problem.setObjectiveN(opt_ratio, 0, 1)
        opt_lp_problem.update()
    else:
        opt_ratio = opt_ratio_value

    for _arch in net_direct.edges:
        _arch_capacity = net_direct.get_edge_key(_arch, key=EdgeConsts.CAPACITY_STR)
        opt_lp_problem.addConstr(sum(arch_all_vars[_arch]), GRB.LESS_EQUAL, _arch_capacity * opt_ratio)

    opt_lp_problem.update()

    for flow in flows:
        src, dst = flow
        # Flow conservation at the source
        from_its_src = sum(arch_vars_per_flow[out_arch][flow] for out_arch in net_direct.out_edges_by_node(src))
        to_its_src = sum(arch_vars_per_flow[in_arch][flow] for in_arch in net_direct.in_edges_by_node(src))
        opt_lp_problem.addConstr(from_its_src - to_its_src, GRB.EQUAL, traffic_matrix[flow],
                                 "{}->{};srcConst".format(src, dst))
        # opt_lp_problem.addConstr(to_its_src, GRB.EQUAL, 0)

        # Flow conservation at the destination
        from_its_dst = sum(arch_vars_per_flow[out_arch][flow] for out_arch in net_direct.out_edges_by_node(dst))
        to_its_dst = sum(arch_vars_per_flow[in_arch][flow] for in_arch in net_direct.in_edges_by_node(dst))
        opt_lp_problem.addConstr(to_its_dst - from_its_dst, GRB.EQUAL, traffic_matrix[flow],
                                 "{}->{};dstConst".format(src, dst))
        # opt_lp_problem.addConstr(from_its_dst, GRB.EQUAL, 0)

        for u in net_direct.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            from_some_u = sum(arch_vars_per_flow[out_arch][flow] for out_arch in net_direct.out_edges_by_node(u))
            to_some_u = sum(arch_vars_per_flow[in_arch][flow] for in_arch in net_direct.in_edges_by_node(u))
            opt_lp_problem.addConstr(from_some_u - to_some_u, GRB.EQUAL, 0, "{}->{};trans_{}_Const".format(src, dst, u))
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

    for _arch in net_direct.edges:
        for flow in flows:
            src, dst = flow
            assert src != dst
            assert traffic_matrix[flow] > 0
            arch_vars_per_flow[_arch][flow] = arch_vars_per_flow[_arch][flow].x

    opt_lp_problem.close()
    __validate_solution(net_direct, flows, traffic_matrix, arch_vars_per_flow)

    link_carries_per_flow = defaultdict(lambda: np.zeros(shape=traffic_matrix.shape, dtype=np.float64))
    for arch in net_direct.edges:
        for flow in flows:
            flow_demand = traffic_matrix[flow]
            link_carries_per_flow[arch][flow] += float(arch_vars_per_flow[arch][flow]) / float(flow_demand)

    max_congested_link = 0

    for link in net_direct.edges:
        link_capacity = net_direct.get_edge_key(link, EdgeConsts.CAPACITY_STR)
        fractions_from_lp = link_carries_per_flow[link]
        total_link_load = np.sum(np.multiply(fractions_from_lp, traffic_matrix))
        max_congested_link = max(max_congested_link, total_link_load / link_capacity)

    assert error_bound(max_congested_link, opt_ratio)
    return max_congested_link, link_carries_per_flow
