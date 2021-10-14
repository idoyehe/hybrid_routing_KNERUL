from common.consts import EdgeConsts, Consts
from common.utils import extract_flows, load_dump_file, error_bound, extract_lp_values
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.logger import *
from sys import argv
import gurobipy as gb
from gurobipy import GRB
from argparse import ArgumentParser
import numpy as np
from random import shuffle
from tabulate import tabulate


def multiple_tms_mcf_LP_solver(net: NetworkClass, traffic_matrix_list):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, Consts.OUTPUT_FLAG)
    gb_env.setParam(GRB.Param.NumericFocus, Consts.NUMERIC_FOCUS)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.start()

    expected_objective, r_per_mtrx, necessary_capacity_per_tm = _aux_multiple_tms_mcf_LP_solver(net, traffic_matrix_list, gb_env)
    while True:
        try:
            expected_objective, r_per_mtrx, necessary_capacity_per_tm = \
                _aux_multiple_tms_mcf_LP_solver(net, traffic_matrix_list, gb_env, expected_objective - 0.001)
            print("****** Gurobi Failure ******")
            expected_objective -= 0.001
        except Exception as e:
            return expected_objective, r_per_mtrx, necessary_capacity_per_tm


def _aux_multiple_tms_mcf_LP_solver(net_direct: NetworkClass, traffic_matrices_list, gurobi_env, expected_objective=None):
    """Preparation"""
    mcf_problem = gb.Model(name="LP expected max congestion, given network topology, Traffic Matrices distribution", env=gurobi_env)
    tms_list_length = len(traffic_matrices_list)
    demands_ratios = np.zeros(shape=(tms_list_length, net_direct.get_num_nodes, net_direct.get_num_nodes))
    aggregate_tm = sum(tm for _, tm in traffic_matrices_list)
    active_flows = extract_flows(aggregate_tm)

    vars_flows_src_dst_per_edge = mcf_problem.addVars(active_flows, net_direct.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)
    vars_bt_per_matrix = mcf_problem.addVars(tms_list_length, name="bt", lb=0.0, vtype=GRB.CONTINUOUS)
    mcf_problem.update()

    """Building Constraints"""
    total_objective = sum(tm_prb * vars_bt_per_matrix[m_idx] for m_idx, (tm_prb, _) in enumerate(traffic_matrices_list))

    if expected_objective is None:
        mcf_problem.setObjective(total_objective, GRB.MINIMIZE)
    else:
        mcf_problem.addLConstr(total_objective, GRB.LESS_EQUAL, expected_objective)

    mcf_problem.update()

    # extracting demands ratios per single matrix
    for m_idx, (_, tm) in enumerate(traffic_matrices_list):
        for src, dst in active_flows:
            assert aggregate_tm[src, dst] > 0
            demands_ratios[m_idx, src, dst] = tm[src, dst] / aggregate_tm[src, dst]
            assert 0 <= demands_ratios[m_idx, src, dst] <= 1

    for u, v in net_direct.edges:
        edge_capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
        relevant_flows = list(filter(lambda src_dst: src_dst[1] != u, active_flows))
        for m_idx in range(tms_list_length):
            m_link_load = sum(vars_flows_src_dst_per_edge[src, dst, u, v] * demands_ratios[m_idx, src, dst] for src, dst in relevant_flows)
            mcf_problem.addLConstr(m_link_load, GRB.LESS_EQUAL, edge_capacity * vars_bt_per_matrix[m_idx])

    for src, dst in active_flows:
        # Flow conservation at the source
        __flow_from_src = sum(vars_flows_src_dst_per_edge[src, dst, src, v] for _, v in net_direct.out_edges_by_node(src))
        __flow_to_src = sum(vars_flows_src_dst_per_edge[src, dst, u, src] for u, _ in net_direct.in_edges_by_node(src))
        mcf_problem.addLConstr(__flow_from_src - __flow_to_src == aggregate_tm[src, dst])

        # Flow conservation at the destination
        __flow_to_dst = sum(vars_flows_src_dst_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
        mcf_problem.addLConstr(__flow_to_dst == aggregate_tm[src, dst])
        mcf_problem.addConstrs((vars_flows_src_dst_per_edge[src, dst, dst, v] == 0.0 for _, v in net_direct.out_edges_by_node(dst)))

        for u in net_direct.nodes:
            if u in (src, dst):
                continue
            # Flow conservation at transit node
            __flow_from_u = sum(vars_flows_src_dst_per_edge[src, dst, u, v_out] for _, v_out in net_direct.out_edges_by_node(u))
            __flow_to_u = sum(vars_flows_src_dst_per_edge[src, dst, v_in, u] for v_in, _ in net_direct.in_edges_by_node(u))
            mcf_problem.addLConstr(__flow_from_u == __flow_to_u)

    mcf_problem.update()
    try:
        logger.info("LP Submit to Solve {}".format(mcf_problem.ModelName))
        mcf_problem.optimize()
        assert mcf_problem.Status == GRB.OPTIMAL
    except AssertionError as e:
        raise Exception("****Optimize failed****\nAssertion Error:\n{}".format(e))

    except gb.GurobiError as e:
        raise Exception("****Optimize failed****\nException is:\n{}".format(e))

    if logger.level == logging.DEBUG:
        mcf_problem.printStats()
        mcf_problem.printQuality()

    if expected_objective is None:
        expected_objective = total_objective.getValue()

    flows_src_dst_per_edge = extract_lp_values(vars_flows_src_dst_per_edge)
    bt_per_mtrx = extract_lp_values(vars_bt_per_matrix)
    mcf_problem.close()

    __validate_solution(net_direct, aggregate_tm, flows_src_dst_per_edge)

    necessary_capacity_per_matrix = np.zeros(shape=(tms_list_length + 1, net_direct.get_num_nodes, net_direct.get_num_nodes))
    for m_idx in range(tms_list_length):
        for u, v in net_direct.edges:
            relevant_flows = list(filter(lambda src_dst: src_dst[1] != u, active_flows))
            necessary_capacity_per_matrix[m_idx, u, v] = sum(
                flows_src_dst_per_edge[src, dst, u, v] * demands_ratios[m_idx, u, v] for src, dst in relevant_flows)

    for u, v in net_direct.edges:
        necessary_capacity_per_matrix[tms_list_length, u, v] = sum(flows_src_dst_per_edge[src, dst, u, v] for src, dst in active_flows)

    return expected_objective, bt_per_mtrx, necessary_capacity_per_matrix


def __validate_solution(net_direct, aggregate_tm, flows_src_dst_per_edge):
    active_flows = extract_flows(aggregate_tm)
    for src, dst in active_flows:
        for u in net_direct.nodes:
            if u == src:
                from_src = sum(flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                to_src = sum(flows_src_dst_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                assert error_bound(from_src - to_src, aggregate_tm[src, dst])

            elif u == dst:
                from_dst = sum(flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(from_dst)
                to_dst = sum(flows_src_dst_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                assert error_bound(to_dst, aggregate_tm[src, dst])

            else:
                assert u not in (src, dst)
                to_transit_u = sum(flows_src_dst_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                from_transit_u = sum(flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(to_transit_u, from_transit_u)


def __create_weighted_traffic_matrices(n, tms_list, probability_distribution=None, shuffling: bool = True):
    if shuffling:
        shuffle(tms_list)

    if probability_distribution is None:
        probability_distribution = [1 / n] * n

    assert error_bound(np.sum(probability_distribution), 1.0)
    return [(probability_distribution[i], t[0]) for i, t in enumerate(tms_list[0:n])]


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    parser.add_argument("-n", "--number_of_matrices", type=int, help="Number of matrices to evaluate", default=1)
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    number_of_matrices = _getOptions().number_of_matrices
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))

    traffic_matrix_list = __create_weighted_traffic_matrices(number_of_matrices, loaded_dict["tms"], shuffling=False)

    expected_objective, bt_per_matrix, necessary_capacity_per_matrix_dict = multiple_tms_mcf_LP_solver(net, traffic_matrix_list)

    data = list()
    headers = ["# Matrix", "Congestion using multiple MCF - LP", "Congestion using LP optimal", "Ratio"]
    new_list = list()
    for idx, t_elem in enumerate(loaded_dict["tms"][0:number_of_matrices]):
        bt_per_matrix[idx] = bt_per_matrix[idx]
        assert bt_per_matrix[idx] >= t_elem[1] or error_bound(bt_per_matrix[idx], t_elem[1])
        data.append([idx, bt_per_matrix[idx], t_elem[1], bt_per_matrix[idx] / t_elem[1]])
        new_list.append((t_elem[0], bt_per_matrix[idx], None))
    print(tabulate(data, headers=headers))
    print("LP Expected Objective :{}".format(expected_objective))
    print("Averaged Congestion :{}".format(np.mean(list(bt_per_matrix.values()))))
