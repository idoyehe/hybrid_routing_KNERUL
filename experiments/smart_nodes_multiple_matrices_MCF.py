import time

from common.consts import EdgeConsts, Consts
from common.utils import change_zero_cells, extract_lp_values
from common.network_class import NetworkClass
from common.logger import *
import gurobipy as gb
from gurobipy import GRB, tupledict
from common.utils import error_bound, extract_flows
import numpy as np
from random import shuffle


def __validate_splitting_ratios(net_direct, smart_nodes, flows_src_dst_per_node, active_flows, spr_src_dst_per_sn_edges):
    for u in smart_nodes:
        reduced_flows = list(filter(lambda src_dst: src_dst[1] != u, active_flows))
        for src, dst in reduced_flows:
            if flows_src_dst_per_node[src, dst, u] > 0:
                assert error_bound(sum(spr_src_dst_per_sn_edges[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u)), 1.0)
                assert all(list(0 <= spr_src_dst_per_sn_edges[src, dst, u, v] <= 1.0 for _, v in net_direct.out_edges_by_node(u)))


def __validate_flows(net_direct, smart_nodes, flows_src_dst_per_node, traffic_matrices_list, demands_ratios, spr_src_dst_per_sn_edges,
                     destination_based_spr):
    for m_index, tm in enumerate(traffic_matrices_list):
        current_matrix_flows_src_dst_per_node = dict()
        for key, val in flows_src_dst_per_node.items():
            current_matrix_flows_src_dst_per_node[key] = val * demands_ratios[m_index, key[0], key[1]]

        __validate_flows_per_matrix(net_direct, smart_nodes, current_matrix_flows_src_dst_per_node, tm, spr_src_dst_per_sn_edges,
                                    destination_based_spr)


def __validate_flows_per_matrix(net_direct, smart_nodes, current_matrix_flows_src_dst_per_node, tm, spr_src_dst_per_sn_edges, destination_based_spr):
    active_flows = extract_flows(tm)
    for src, dst in active_flows:
        assert error_bound(current_matrix_flows_src_dst_per_node[src, dst, dst], tm[src, dst])
        for v in net_direct.nodes:
            _flow_to_node = 0
            for u, _ in net_direct.in_edges_by_node(v):
                if u == dst:
                    continue
                if u in smart_nodes and current_matrix_flows_src_dst_per_node[src, dst, u] > 0:
                    _flow_to_node += current_matrix_flows_src_dst_per_node[src, dst, u] * spr_src_dst_per_sn_edges[src, dst, u, v]
                else:
                    _edge_idx = net_direct.get_edge2id(u, v)
                    _flow_to_node += current_matrix_flows_src_dst_per_node[src, dst, u] * destination_based_spr[dst][_edge_idx]
            if v == src:
                assert error_bound(_flow_to_node + tm[src, dst], current_matrix_flows_src_dst_per_node[src, dst, v])
            else:
                assert error_bound(_flow_to_node, current_matrix_flows_src_dst_per_node[src, dst, v])


def _aux_mcf_LP_with_smart_nodes_solver(gurobi_env, net_direct: NetworkClass,
                                        traffic_matrices_list, destination_based_spr,
                                        smart_nodes, expected_objective=None):
    mcf_problem = gb.Model(name="Multiple Matrices MCF Problem Finding Smart Nodes Source-Target Routing Scheme", env=gurobi_env)

    tms_list_length = len(traffic_matrices_list)
    demands_ratios = np.zeros_like(traffic_matrices_list)
    aggregate_tm = np.sum(traffic_matrices_list, axis=0)
    active_flows = extract_flows(aggregate_tm)

    # extracting demands ratios per single matrix
    for m_idx, tm in enumerate(traffic_matrices_list):
        for src, dst in active_flows:
            assert aggregate_tm[src, dst] > 0
            demands_ratios[m_idx, src, dst] = tm[src, dst] / aggregate_tm[src, dst]
            assert 0 <= demands_ratios[m_idx, src, dst] <= 1

    vars_flows_src_dst_per_node = mcf_problem.addVars(active_flows, net_direct.nodes, name="f", lb=0, vtype=GRB.CONTINUOUS)
    vars_bt_per_matrix = mcf_problem.addVars(tms_list_length, name="bt", lb=0, vtype=GRB.CONTINUOUS)

    mcf_problem.update()
    """form objective"""
    total_objective = (1 / tms_list_length) * vars_bt_per_matrix.sum()

    if expected_objective is None:
        mcf_problem.setObjective(total_objective, GRB.MINIMIZE)
    else:
        mcf_problem.addLConstr(total_objective, GRB.LESS_EQUAL, expected_objective)

    mcf_problem.update()

    # building smart nodes constraints
    vars_flows_src_dst_per_sn_edges = tupledict()
    for s_n in smart_nodes:
        reduced_flows = list(filter(lambda src_dst: src_dst[1] != s_n, active_flows))  # exclude flows the smart node is the destination
        vars_flows_src_dst_per_sn_edges.update(mcf_problem.addVars(reduced_flows, net_direct.out_edges_by_node(s_n), name="f_sn", lb=0,
                                                                   vtype=GRB.CONTINUOUS))
        for src, dst in reduced_flows:
            out_flow_per_src_dst = sum(vars_flows_src_dst_per_sn_edges[src, dst, s_n, v] for _, v in net_direct.out_edges_by_node(s_n))
            mcf_problem.addLConstr(vars_flows_src_dst_per_node[src, dst, s_n], GRB.EQUAL, out_flow_per_src_dst)
    mcf_problem.update()

    for u, v in net_direct.edges:
        edge_capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
        _edge_idx = net_direct.get_edge2id(u, v)
        reduced_flows = list(filter(lambda src_dst: src_dst[1] != u, active_flows))
        for m_idx in range(tms_list_length):
            if u in smart_nodes:
                m_link_load = sum(vars_flows_src_dst_per_sn_edges[src, dst, u, v] * demands_ratios[m_idx, src, dst] for src, dst in reduced_flows)
            else:
                m_link_load = sum(vars_flows_src_dst_per_node[src, dst, u] * demands_ratios[m_idx, src, dst] * destination_based_spr[dst, _edge_idx]
                                  for src, dst in reduced_flows)
            mcf_problem.addLConstr(m_link_load, GRB.LESS_EQUAL, edge_capacity * vars_bt_per_matrix[m_idx])

    for src, dst in active_flows:
        # Flow conservation at the dst
        mcf_problem.addLConstr(vars_flows_src_dst_per_node[src, dst, dst], GRB.EQUAL, aggregate_tm[src, dst])
        for v in net_direct.nodes:
            _flow_to_node = 0
            for u, _ in net_direct.in_edges_by_node(v):
                if u == dst:
                    continue
                if u in smart_nodes:
                    _flow_to_node += vars_flows_src_dst_per_sn_edges[src, dst, u, v]
                else:
                    _edge_idx = net_direct.get_edge2id(u, v)
                    _flow_to_node += vars_flows_src_dst_per_node[src, dst, u] * destination_based_spr[dst][_edge_idx]

            if v == src:
                mcf_problem.addLConstr(_flow_to_node + aggregate_tm[src, dst], GRB.EQUAL, vars_flows_src_dst_per_node[src, dst, v])
            else:
                mcf_problem.addLConstr(_flow_to_node, GRB.EQUAL, vars_flows_src_dst_per_node[src, dst, v])

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

    if logger.level == logging.DEBUG:
        mcf_problem.printStats()
        mcf_problem.printQuality()

    if expected_objective is None:
        expected_objective = total_objective.getValue()

    flows_src_dst_per_node = extract_lp_values(vars_flows_src_dst_per_node, Consts.ROUND)
    flows_src_dst_per_sn_edges = extract_lp_values(vars_flows_src_dst_per_sn_edges, Consts.ROUND)
    bt_per_matrix = extract_lp_values(vars_bt_per_matrix, Consts.ROUND)
    mcf_problem.close()

    del vars_flows_src_dst_per_node
    del vars_flows_src_dst_per_sn_edges
    del vars_bt_per_matrix

    spr_src_dst_per_sn_edges = dict()
    for u in smart_nodes:
        reduced_flows = list(filter(lambda src_dst: src_dst[1] != u, active_flows))
        for src, dst in reduced_flows:
            flow_from_u_src2dst = flows_src_dst_per_node[src, dst, u]
            if flow_from_u_src2dst == 0:
                continue
            assert len(net_direct.out_edges_by_node(u)) > 1
            for _, v in net_direct.out_edges_by_node(u):
                spr_src_dst_per_sn_edges[(src, dst, u, v)] = flows_src_dst_per_sn_edges[src, dst, u, v] / flow_from_u_src2dst

    __validate_splitting_ratios(net_direct, smart_nodes, flows_src_dst_per_node, active_flows, spr_src_dst_per_sn_edges)
    __validate_flows(net_direct, smart_nodes, flows_src_dst_per_node, traffic_matrices_list, demands_ratios, spr_src_dst_per_sn_edges,
                     destination_based_spr)

    return expected_objective, spr_src_dst_per_sn_edges


def matrices_mcf_LP_with_smart_nodes_solver(smart_nodes, net: NetworkClass, traffic_matrix_list, destination_based_spr):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, Consts.NUMERIC_FOCUS)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.start()

    expected_objective, splitting_ratios_per_src_dst_edge = \
        _aux_mcf_LP_with_smart_nodes_solver(gb_env, net, traffic_matrix_list, destination_based_spr, smart_nodes)
    return expected_objective, smart_nodes, splitting_ratios_per_src_dst_edge


def create_random_TMs_list(length, traffic_matrices_list, shuffling: bool = True):
    if shuffling:
        shuffle(traffic_matrices_list)
    return np.array([t[0] for t in traffic_matrices_list[0:length]])