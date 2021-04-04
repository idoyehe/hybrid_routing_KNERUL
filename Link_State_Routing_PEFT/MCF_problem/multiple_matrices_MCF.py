from common.consts import EdgeConsts
from common.utils import change_zero_cells
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.logger import *
from sys import argv
import gurobipy as gb
from gurobipy import GRB
from argparse import ArgumentParser
from common.utils import load_dump_file, error_bound, extract_flows
import numpy as np
from random import shuffle
from tabulate import tabulate
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer

R = 10


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


def __validate_flow_per_matrix(net_direct, tm, flows_per_edge_src_dst, splitting_ratios_per_src_dst_edge):
    current_flows = extract_flows(tm)

    for src, dst in current_flows:
        for u in net_direct.nodes:
            if u == src:
                from_src = sum(flows_per_edge_src_dst[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(from_src, tm[src, dst])
                to_src = sum(flows_per_edge_src_dst[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                assert error_bound(to_src)

                for _, v in net_direct.out_edges_by_node(u):
                    assert error_bound(flows_per_edge_src_dst[src, dst, u, v],
                                       from_src * splitting_ratios_per_src_dst_edge[
                                           src, dst, u, v])

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


def __validate_flow(net_direct, traffic_matrix_list, flows_per_mtrx_src_dst_per_edge,
                    splitting_ratios_edge_src_dst):
    for m_index, (_, tm) in enumerate(traffic_matrix_list):
        current_matrix_flows_per_src_dst_per_edge = dict()
        for key, val in flows_per_mtrx_src_dst_per_edge.items():
            if key[0] == m_index:
                current_matrix_flows_per_src_dst_per_edge[key[1:]] = val

        __validate_flow_per_matrix(net_direct, tm, current_matrix_flows_per_src_dst_per_edge,
                                   splitting_ratios_edge_src_dst)


def __validate_solution(net_direct, flows, traffic_matrix_list, splitting_ratios_per_src_dst_edge,
                        flows_per_mtrx_src_dst_per_edge):
    __validate_splitting_ratios(net_direct, flows, splitting_ratios_per_src_dst_edge)

    __validate_flow(net_direct, traffic_matrix_list, flows_per_mtrx_src_dst_per_edge,
                    splitting_ratios_per_src_dst_edge)


def __extract_values(gurobi_vars_dict):
    gurobi_vars_dict = dict(gurobi_vars_dict)

    for key in gurobi_vars_dict.keys():
        gurobi_vars_dict[key] = round(gurobi_vars_dict[key].x, R)
    return gurobi_vars_dict


def _aux_mcf_LP_expected_matrix_heuristic_solver(net_direct: NetworkClass, traffic_matrices_list, gurobi_env,
                                                 opt_ratio_value=None):
    """Preparation"""
    mcf_problem = gb.Model(name="MCF problem for mean MCF, given network, TM list and probabilities",
                           env=gurobi_env)

    expected_demands = sum(pr * t for pr, t in traffic_matrices_list)

    flows = extract_flows(expected_demands)

    vars_flows_src_dst_per_edge = mcf_problem.addVars(flows, net_direct.edges,
                                                      name="f", lb=0.0, vtype=GRB.CONTINUOUS)

    vars_r_total_matrix = mcf_problem.addVar(name="r_total", lb=0.0, vtype=GRB.CONTINUOUS)

    mcf_problem.update()
    """Building Constraints"""
    total_objective = vars_r_total_matrix

    if opt_ratio_value is None:
        mcf_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        mcf_problem.setObjective(total_objective)
    else:
        mcf_problem.addConstr(total_objective <= opt_ratio_value)
    mcf_problem.update()

    for u, v in net_direct.edges:
        capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
        mtrx_link_load = sum(vars_flows_src_dst_per_edge[src, dst, u, v] for src, dst in flows)
        mcf_problem.addConstr(mtrx_link_load <= capacity * vars_r_total_matrix)

    for src, dst in flows:
        # Flow conservation at the source
        __flow_from_src = sum(
            vars_flows_src_dst_per_edge[src, dst, src, v] for _, v in net_direct.out_edges_by_node(src))
        __flow_to_src = sum(
            vars_flows_src_dst_per_edge[src, dst, u, src] for u, _ in net_direct.in_edges_by_node(src))
        mcf_problem.addConstr(__flow_from_src == expected_demands[src, dst])
        mcf_problem.addConstr(__flow_to_src == 0.0)

        # Flow conservation at the destination
        __flow_from_dst = sum(
            vars_flows_src_dst_per_edge[src, dst, dst, v] for _, v in net_direct.out_edges_by_node(dst))
        __flow_to_dst = sum(
            vars_flows_src_dst_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
        mcf_problem.addConstr(__flow_to_dst == expected_demands[src, dst])
        mcf_problem.addConstr(__flow_from_dst == 0.0)

        for u in net_direct.nodes:
            if u in (src, dst):
                continue
            # Flow conservation at transit node
            __flow_from_u = sum(
                vars_flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            __flow_to_u = sum(
                vars_flows_src_dst_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
            mcf_problem.addConstr(__flow_from_u == __flow_to_u)
        mcf_problem.update()

    try:
        logger.info("LP Submit to Solve {}".format(mcf_problem.ModelName))
        mcf_problem.update()
        mcf_problem.optimize()
        assert mcf_problem.Status == GRB.OPTIMAL
    except AssertionError as e:
        raise Exception("****Optimize failed****\nAssertion Error:\n{}".format(e))

    except gb.GurobiError as e:
        raise Exception("****Optimize failed****\nException is:\n{}".format(e))

    if opt_ratio_value is None:
        opt_ratio_value = round(mcf_problem.objVal, R)

    if logger.level == logging.DEBUG:
        mcf_problem.printStats()
        mcf_problem.printQuality()

    flows_src_dst_per_edge = __extract_values(vars_flows_src_dst_per_edge)
    heuristic_optimal = round(vars_r_total_matrix.x, R)
    assert heuristic_optimal == opt_ratio_value
    mcf_problem.close()

    splitting_ratios_per_src_dst_edge = dict()
    for u in net_direct.nodes:
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            flow_from_u_to_dst = sum(
                flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            if flow_from_u_to_dst > 0:
                for _, v in net_direct.out_edges_by_node(u):
                    splitting_ratios_per_src_dst_edge[src, dst, u, v] = flows_src_dst_per_edge[
                                                                            src, dst, u, v] / flow_from_u_to_dst
            else:
                equal_splitting_ratio = 1 / len(net_direct.out_edges_by_node(u))
                for _, v in net_direct.out_edges_by_node(u):
                    splitting_ratios_per_src_dst_edge[src, dst, u, v] = equal_splitting_ratio

    __validate_splitting_ratios(net_direct, flows, splitting_ratios_per_src_dst_edge)
    __validate_flow_per_matrix(net_direct, expected_demands, flows_src_dst_per_edge, splitting_ratios_per_src_dst_edge)

    necessary_capacity_dict = dict()
    for u, v in net_direct.edges:
        necessary_capacity_dict[u, v] = 0
        for src, dst in flows:
            necessary_capacity_dict[u, v] += flows_src_dst_per_edge[src, dst, u, v]

    return heuristic_optimal, splitting_ratios_per_src_dst_edge, necessary_capacity_dict


def _aux_mcf_LP_baseline_solver(net_direct: NetworkClass, traffic_matrices_list, gurobi_env, expected_objective=None):
    """Preparation"""
    mcf_problem = gb.Model(name="MCF problem for mean MCF, given network, TM list and probabilities",
                           env=gurobi_env)

    traffic_matrices_list_length = len(traffic_matrices_list)

    total_demands = sum(t for pr, t in traffic_matrices_list)

    flows = extract_flows(total_demands)

    vars_flows_src_dst_per_edge = mcf_problem.addVars(flows, net_direct.edges,
                                                      name="f", lb=0.0, vtype=GRB.CONTINUOUS)

    vars_flows_per_mtrx_src_dst_per_edge = mcf_problem.addVars(traffic_matrices_list_length,
                                                               flows, net_direct.edges,
                                                               name="f_m", lb=0.0, vtype=GRB.CONTINUOUS)

    vars_r_per_mtrx = mcf_problem.addVars(traffic_matrices_list_length, name="r", lb=0.0, vtype=GRB.CONTINUOUS)

    mcf_problem.update()
    """Building Constraints"""
    total_objective = sum(tm_prb * vars_r_per_mtrx[m_idx] for m_idx, (tm_prb, _) in enumerate(traffic_matrices_list))

    if expected_objective is None:
        mcf_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        mcf_problem.setObjective(total_objective)
    else:
        mcf_problem.addConstr(total_objective <= expected_objective)
    mcf_problem.update()

    # extracting s,t flow carried by (u,v) per matrix
    for src, dst in flows:
        for mtrx_idx, (_, tm) in enumerate(traffic_matrices_list):
            assert total_demands[src, dst] > 0
            demand_ratio = tm[src, dst] / total_demands[src, dst]
            assert demand_ratio >= 0 and demand_ratio <= 1
            mcf_problem.addConstrs(vars_flows_per_mtrx_src_dst_per_edge[mtrx_idx, src, dst, u, v] ==
                                   demand_ratio * vars_flows_src_dst_per_edge[src, dst, u, v]
                                   for u, v in net_direct.edges)

    for mtrx_idx in range(traffic_matrices_list_length):
        for u, v in net_direct.edges:
            capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
            mtrx_link_load = sum(
                vars_flows_per_mtrx_src_dst_per_edge[mtrx_idx, src, dst, u, v] for src, dst in flows)
            mcf_problem.addConstr(mtrx_link_load <= capacity * vars_r_per_mtrx[mtrx_idx])

    for src, dst in flows:
        # Flow conservation at the source
        __flow_from_src = sum(
            vars_flows_src_dst_per_edge[src, dst, src, v] for _, v in net_direct.out_edges_by_node(src))
        __flow_to_src = sum(
            vars_flows_src_dst_per_edge[src, dst, u, src] for u, _ in net_direct.in_edges_by_node(src))
        mcf_problem.addConstr(__flow_from_src == total_demands[src, dst])
        mcf_problem.addConstr(__flow_to_src == 0.0)

        # Flow conservation at the destination
        __flow_from_dst = sum(
            vars_flows_src_dst_per_edge[src, dst, dst, v] for _, v in net_direct.out_edges_by_node(dst))
        __flow_to_dst = sum(
            vars_flows_src_dst_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
        mcf_problem.addConstr(__flow_to_dst == total_demands[src, dst])
        mcf_problem.addConstr(__flow_from_dst == 0.0)

        for u in net_direct.nodes:
            if u in (src, dst):
                continue
            # Flow conservation at transit node
            __flow_from_u = sum(
                vars_flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            __flow_to_u = sum(
                vars_flows_src_dst_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
            mcf_problem.addConstr(__flow_from_u == __flow_to_u)
        mcf_problem.update()

    try:
        logger.info("LP Submit to Solve {}".format(mcf_problem.ModelName))
        mcf_problem.update()
        mcf_problem.optimize()
        assert mcf_problem.Status == GRB.OPTIMAL
    except AssertionError as e:
        raise Exception("****Optimize failed****\nAssertion Error:\n{}".format(e))

    except gb.GurobiError as e:
        raise Exception("****Optimize failed****\nException is:\n{}".format(e))

    if expected_objective is None:
        expected_objective = round(mcf_problem.objVal, R)

    if logger.level == logging.DEBUG:
        mcf_problem.printStats()
        mcf_problem.printQuality()

    flows_src_dst_per_edge = __extract_values(vars_flows_src_dst_per_edge)
    flows_per_mtrx_src_dst_per_edge = __extract_values(vars_flows_per_mtrx_src_dst_per_edge)
    r_per_mtrx = __extract_values(vars_r_per_mtrx)
    mcf_problem.close()

    splitting_ratios_per_src_dst_edge = dict()
    for u in net_direct.nodes:
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            flow_from_u_to_dst = sum(flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            if flow_from_u_to_dst > 0:
                for _, v in net_direct.out_edges_by_node(u):
                    splitting_ratios_per_src_dst_edge[src, dst, u, v] = flows_src_dst_per_edge[src, dst, u, v] / flow_from_u_to_dst
            else:
                for _, v in net_direct.out_edges_by_node(u):
                    splitting_ratios_per_src_dst_edge[src, dst, u, v] = None

    __validate_solution(net_direct, flows, traffic_matrices_list, splitting_ratios_per_src_dst_edge,
                        flows_per_mtrx_src_dst_per_edge)

    necessary_capacity_per_matrix_dict = dict()
    for m_index in range(traffic_matrices_list_length):
        for u, v in net_direct.edges:
            necessary_capacity = 0
            for src, dst in flows:
                necessary_capacity += flows_per_mtrx_src_dst_per_edge[m_index, src, dst, u, v]
            necessary_capacity_per_matrix_dict[m_index, u, v] = necessary_capacity

    return expected_objective, splitting_ratios_per_src_dst_edge, r_per_mtrx, necessary_capacity_per_matrix_dict


def multiple_matrices_mcf_LP_baseline_solver(net: NetworkClass, traffic_matrix_list):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 3)
    gb_env.setParam(GRB.Param.FeasibilityTol, 1e-9)
    gb_env.start()

    expected_objective, splitting_ratios_per_src_dst_edge, r_per_mtrx, necessary_capacity_per_matrix_dict = \
        _aux_mcf_LP_baseline_solver(net, traffic_matrix_list, gb_env)
    while True:
        try:
            expected_objective, splitting_ratios_per_src_dst_edge, r_per_mtrx, necessary_capacity_per_matrix_dict = \
                _aux_mcf_LP_baseline_solver(net, traffic_matrix_list, gb_env, expected_objective - 0.001)
            print("****** Gurobi Failure ******")
            expected_objective -= 0.001
        except Exception as e:
            return expected_objective, splitting_ratios_per_src_dst_edge, r_per_mtrx, necessary_capacity_per_matrix_dict


def multiple_matrices_mcf_LP_heuristic_solver(net: NetworkClass, traffic_matrix_list):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 3)
    gb_env.setParam(GRB.Param.FeasibilityTol, 1e-9)
    gb_env.start()

    heuristic_optimal, splitting_ratios_per_src_dst_edge, necessary_capacity_dict = \
        _aux_mcf_LP_expected_matrix_heuristic_solver(net, traffic_matrix_list, gb_env)
    while True:
        try:
            heuristic_optimal, splitting_ratios_per_src_dst_edge, necessary_capacity_dict = \
                _aux_mcf_LP_expected_matrix_heuristic_solver(net, traffic_matrix_list, gb_env,
                                                             heuristic_optimal - 0.001)
            print("****** Gurobi Failure ******")
            heuristic_optimal -= 0.001
        except Exception as e:
            return heuristic_optimal, splitting_ratios_per_src_dst_edge, necessary_capacity_dict


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


def weighted_destination_routing(net_direct: NetworkClass, weight_matrix, splitting_ratios_per_src_dst_edge):
    splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)
    for t in net_direct.nodes:
        for u, v in net_direct.edges:
            edge_idx = net_direct.get_edge2id(u, v)
            normalized = 0
            weighted_sum = 0
            for s in net_direct.nodes:
                if s == t:
                    continue
                assert (s, t, u, v) in splitting_ratios_per_src_dst_edge.keys()
                if splitting_ratios_per_src_dst_edge[s, t, u, v] is not None:
                    weighted_sum += weight_matrix[s, t] * splitting_ratios_per_src_dst_edge[s, t, u, v]
                    normalized += weight_matrix[s, t]
            if normalized > 0:
                splitting_ratios[t, edge_idx] = weighted_sum / normalized
            else:
                splitting_ratios[t, edge_idx] = None

    for t in net_direct.nodes:
        for u in net_direct.nodes:
            if u == t:
                assert all(np.isnan(splitting_ratios[t, net_direct.get_edge2id(u, v)]) for _, v in net_direct.out_edges_by_node(u))
            else:
                assert error_bound(sum(splitting_ratios[t, net_direct.get_edge2id(u, v)] for _, v in net_direct.out_edges_by_node(u)), 1.0)

    return np.nan_to_num(splitting_ratios)


def __create_weighted_traffic_matrices(length, traffic_matrix_list, probability_distribution, shuffling: bool = True):
    assert error_bound(np.sum(probability_distribution), 1.0)
    if shuffling:
        shuffle(traffic_matrix_list)
    return [(probability_distribution[i], change_zero_cells(t[0])) for i, t in enumerate(traffic_matrix_list[0:length])]


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))

    length = 500
    # pr = [0.99] + [(1 - 0.99) / (length - 1)] * (length - 1)
    pr = [1 / length] * length

    traffic_matrix_list = __create_weighted_traffic_matrices(length, loaded_dict["tms"], pr)

    expected_objective, splitting_ratios_per_src_dst_edge, r_vars_per_matrix, necessary_capacity_per_matrix_dict = \
        multiple_matrices_mcf_LP_baseline_solver(net, traffic_matrix_list)

    data = list()
    headers = ["# Matrix", "Congestion using multiple MCF - LP", "Congestion using LP optimal", "Ratio"]
    for idx, t_elem in enumerate(loaded_dict["tms"][0:length]):
        r_vars_per_matrix[idx] = round(r_vars_per_matrix[idx], 4)
        assert r_vars_per_matrix[idx] >= t_elem[1] or error_bound(r_vars_per_matrix[idx], t_elem[1])

        data.append([idx, r_vars_per_matrix[idx], t_elem[1], r_vars_per_matrix[idx] / t_elem[1]])

    print(tabulate(data, headers=headers))

    splitting_ratios = weighted_destination_routing(net, sum(pr*t for pr, t in traffic_matrix_list), splitting_ratios_per_src_dst_edge)

    traffic_distribution = PEFTOptimizer(net, None)

    data = list()
    headers = ["# Matrix",
               "Congestion using LP optimal",
               "Congestion using heuristic destination routing",
               "Heuristic Ratio Vs. LP optimal"]

    heuristic_expected_objective = 0
    expected_cost = 0
    length = len(loaded_dict["tms"])
    for idx, t_elem in enumerate(loaded_dict["tms"]):
        max_congestion, _, _, _, _ = traffic_distribution._calculating_traffic_distribution(splitting_ratios, t_elem[0])
        data.append([idx, t_elem[1], max_congestion, max_congestion / t_elem[1]])
        heuristic_expected_objective += 1/length * max_congestion
        expected_cost += 1/length * (max_congestion / t_elem[1])

    print(tabulate(data, headers=headers))
    print("Heuristic expected objective: {}".format(heuristic_expected_objective))
    print("Optimal expected objective: {}".format(expected_objective))
    print("Expected cost: {}".format(expected_cost))
