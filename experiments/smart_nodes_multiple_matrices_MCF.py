from common.consts import EdgeConsts, Consts
from common.utils import change_zero_cells, extract_lp_values
from common.network_class import NetworkClass
from common.logger import *
import gurobipy as gb
from gurobipy import GRB
from common.utils import error_bound, extract_flows
import numpy as np
from random import shuffle


def __validate_splitting_ratios(net_direct, flows, splitting_ratios_per_src_dst_edge):
    for u in net_direct.nodes():
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            splitting_ratios_per_src_dst_u_list = [splitting_ratios_per_src_dst_edge[src, dst, net_direct.get_edge2id(u, v)] for _, v in
                                                   net_direct.out_edges_by_node(u)]
            if all(np.isnan(spt) for spt in splitting_ratios_per_src_dst_u_list):
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
                to_src = sum(flows_per_edge_src_dst[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                assert error_bound(from_src, to_src + tm[src, dst])

                for _, v in net_direct.out_edges_by_node(u):
                    assert error_bound(flows_per_edge_src_dst[src, dst, u, v],
                                       from_src * splitting_ratios_per_src_dst_edge[src, dst, net_direct.get_edge2id(u, v)])

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
                        assert error_bound(flows_per_edge_src_dst[src, dst, u, v],
                                           from_transit_u * splitting_ratios_per_src_dst_edge[src, dst, net_direct.get_edge2id(u, v)])

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


def _aux_mcf_LP_with_smart_nodes_solver(gurobi_env, net_direct: NetworkClass,
                                        traffic_matrices_list, constant_spr, smart_nodes,
                                        expected_objective=None):
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
        # Flow conservation at the dst
        __flow_from_dst = sum(vars_flows_src_dst_per_edge[src, dst, dst, v] for _, v in net_direct.out_edges_by_node(dst))
        __flow_to_dst = sum(vars_flows_src_dst_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
        mcf_problem.addConstr(__flow_to_dst == total_demands[src, dst])
        mcf_problem.addConstr(__flow_from_dst == 0.0)

        for u in net_direct.nodes:
            if u == dst:
                continue
            # Flow conservation at src / transit node
            __flow_from_u = sum(vars_flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            __flow_to_u = sum(vars_flows_src_dst_per_edge[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
            if u == src:
                mcf_problem.addConstr(__flow_from_u == __flow_to_u + total_demands[src, dst])
            else:
                mcf_problem.addConstr(__flow_from_u == __flow_to_u)

            for _u, v in net_direct.out_edges_by_node(u):
                assert u == _u
                del _u
                u_v_idx = net_direct.get_edge2id(u, v)
                if u not in smart_nodes:
                    spr = constant_spr[dst, u_v_idx]
                    mcf_problem.addConstr(__flow_from_u * spr == vars_flows_src_dst_per_edge[src, dst, u, v])

        mcf_problem.update()

    try:
        logger.info("LP Submit to Solve {}".format(mcf_problem.ModelName))
        mcf_problem.update()
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

    if expected_objective is None:
        expected_objective = round(mcf_problem.objVal, Consts.ROUND)

    if logger.level == logging.DEBUG:
        mcf_problem.printStats()
        mcf_problem.printQuality()

    flows_src_dst_per_edge = extract_lp_values(vars_flows_src_dst_per_edge, Consts.ROUND)
    flows_per_mtrx_src_dst_per_edge = extract_lp_values(vars_flows_per_mtrx_src_dst_per_edge, Consts.ROUND)
    for key, value in flows_src_dst_per_edge.items():
        if flows_src_dst_per_edge[key] < 0:
            print(key)

    mcf_problem.close()

    splitting_ratios_per_src_dst_edge = np.empty(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_edges))
    splitting_ratios_per_src_dst_edge.fill(np.nan)
    for u in net_direct.nodes:
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            flow_from_u_src2dst = sum(flows_src_dst_per_edge[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
            if flow_from_u_src2dst > 0:
                for _, v in net_direct.out_edges_by_node(u):
                    edge_idx = net_direct.get_edge2id(u, v)
                    splitting_ratios_per_src_dst_edge[src, dst, edge_idx] = flows_src_dst_per_edge[src, dst, u, v] / flow_from_u_src2dst

    __validate_solution(net_direct, flows, traffic_matrices_list, splitting_ratios_per_src_dst_edge,
                        flows_per_mtrx_src_dst_per_edge)

    return expected_objective, splitting_ratios_per_src_dst_edge


def matrices_mcf_LP_with_smart_nodes_solver(net: NetworkClass, traffic_matrix_list, constant_spr, smart_nodes):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 3)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.start()

    expected_objective, splitting_ratios_per_src_dst_edge = \
        _aux_mcf_LP_with_smart_nodes_solver(gb_env, net, traffic_matrix_list, constant_spr, smart_nodes)
    return expected_objective, smart_nodes, splitting_ratios_per_src_dst_edge


def create_weighted_traffic_matrices(length, traffic_matrix_list, probability_distribution=None, shuffling: bool = True):
    if probability_distribution is None:
        probability_distribution = [1 / length] * length
    assert error_bound(np.sum(probability_distribution), 1.0)
    if shuffling:
        shuffle(traffic_matrix_list)
    return [(probability_distribution[i], change_zero_cells(t[0])) for i, t in enumerate(traffic_matrix_list[0:length])]
