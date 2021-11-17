from common.consts import EdgeConsts, Consts
from common.utils import extract_lp_values
from common.network_class import NetworkClass
from common.logger import *
import gurobipy as gb
from gurobipy import GRB, tupledict
from common.utils import error_bound, extract_flows
import numpy as np


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
                    _flow_to_node += current_matrix_flows_src_dst_per_node[src, dst, u] * destination_based_spr[dst][u, v]
            if v == src:
                assert error_bound(_flow_to_node + tm[src, dst], current_matrix_flows_src_dst_per_node[src, dst, v])
            else:
                assert error_bound(_flow_to_node, current_matrix_flows_src_dst_per_node[src, dst, v])


def _aux_mcf_LP_with_smart_nodes_solver(gurobi_env, net_direct: NetworkClass,
                                        traffic_matrices_list, destination_based_spr,
                                        smart_nodes, expected_objective=None):
    mcf_problem = gb.Model(name="Multiple Matrices MCF Problem Finding Smart Nodes {} Source-Target Routing Scheme".format(smart_nodes),
                           env=gurobi_env)

    tms_list_length = len(traffic_matrices_list)
    demands_ratios = np.zeros_like(traffic_matrices_list, dtype=np.float64)
    aggregate_tm = np.sum(traffic_matrices_list, axis=0)
    active_flows = extract_flows(aggregate_tm)
    tm_prob = 1 / tms_list_length
    # extracting demands ratios per single matrix
    for m_idx, tm in enumerate(traffic_matrices_list):
        for src, dst in active_flows:
            assert aggregate_tm[src, dst] > 0
            demands_ratios[m_idx, src, dst] = tm[src, dst] / aggregate_tm[src, dst]
            assert 0 <= demands_ratios[m_idx, src, dst] <= 1

    vars_flows_src_dst_per_edge = mcf_problem.addVars(active_flows, net_direct.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)
    vars_bt_per_matrix = mcf_problem.addVars(tms_list_length, name="bt", lb=0.0, vtype=GRB.CONTINUOUS)

    mcf_problem.update()
    """form objective"""
    total_objective = vars_bt_per_matrix.sum() * tm_prob

    if expected_objective is None:
        mcf_problem.setObjective(total_objective, GRB.MINIMIZE)
    else:
        mcf_problem.addLConstr(total_objective, GRB.LESS_EQUAL, expected_objective)

    mcf_problem.update()

    for m_idx, tm in enumerate(traffic_matrices_list):
        tms_active_flows = extract_flows(tm)
        for u, v in net_direct.edges:
            edge_capacity = net_direct.get_edge_key((u, v), EdgeConsts.CAPACITY_STR)
            relevant_flows = list(filter(lambda src_dst: src_dst[1] != u, tms_active_flows))
            m_link_load = sum(vars_flows_src_dst_per_edge[src, dst, u, v] * demands_ratios[m_idx, src, dst] for src, dst in relevant_flows)
            mcf_problem.addLConstr(m_link_load, GRB.LESS_EQUAL, edge_capacity * vars_bt_per_matrix[m_idx])

    for node in net_direct.nodes:
        for src, dst in active_flows:
            if node == dst:
                # Flow conservation at the dst
                in_flow_to_dest = sum(vars_flows_src_dst_per_edge[src, dst, u, dst] for u, _ in net_direct.in_edges_by_node(dst))
                mcf_problem.addLConstr(in_flow_to_dest, GRB.EQUAL, aggregate_tm[src, dst])
                mcf_problem.addConstrs((vars_flows_src_dst_per_edge[src, dst, dst, u] == 0.0 for _, u in net_direct.out_edges_by_node(dst)))

            else:
                in_flow_to_node = sum(vars_flows_src_dst_per_edge[src, dst, u, node] for u, _ in net_direct.in_edges_by_node(node))
                if node == src:
                    in_flow_to_node += aggregate_tm[src, dst]
                out_flow_to_node = sum(vars_flows_src_dst_per_edge[src, dst, node, u] for _, u in net_direct.out_edges_by_node(node))
                mcf_problem.addLConstr(out_flow_to_node, GRB.EQUAL, in_flow_to_node)
                if node not in smart_nodes:
                    for _, u in net_direct.out_edges_by_node(node):
                        mcf_problem.addLConstr(in_flow_to_node * destination_based_spr[dst, node, u], GRB.EQUAL,
                                               vars_flows_src_dst_per_edge[src, dst, node, u])

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
        logger.info("Gurobi Model Infeasible, smart node {}".format(smart_nodes))
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

    flows_src_dst_per_edge = extract_lp_values(vars_flows_src_dst_per_edge, Consts.ROUND)
    bt_per_matrix = extract_lp_values(vars_bt_per_matrix, Consts.ROUND)
    mcf_problem.close()

    del vars_flows_src_dst_per_edge
    del vars_bt_per_matrix
    flows_src_dst_per_node = np.zeros(shape=(net_direct.get_num_nodes, net_direct.get_num_nodes, net_direct.get_num_nodes), dtype=np.float64)
    for node in net_direct.nodes:
        for src, dst in active_flows:
            flows_src_dst_per_node[src, dst, node] = sum(flows_src_dst_per_edge[src, dst, node, v] for _, v in net_direct.out_edges_by_node(node))

    spr_src_dst_per_sn_edges = dict()
    for u in smart_nodes:
        reduced_flows = list(filter(lambda src_dst: src_dst[1] != u, active_flows))
        for src, dst in reduced_flows:
            flow_from_u_src2dst = flows_src_dst_per_node[src, dst, u]
            if flow_from_u_src2dst == 0:
                continue
            assert len(net_direct.out_edges_by_node(u)) > 1
            spr_u_v_normalizer = 0
            for _, v in net_direct.out_edges_by_node(u):
                flows_src_dst_per_edge[src, dst, u, v] = max(flows_src_dst_per_edge[src, dst, u, v], 0.0)
                spr_src_dst_per_sn_edges[(src, dst, u, v)] = np.round(flows_src_dst_per_edge[src, dst, u, v] / flow_from_u_src2dst, 4)
                spr_u_v_normalizer += spr_src_dst_per_sn_edges[(src, dst, u, v)]
            for _, v in net_direct.out_edges_by_node(u):
                spr_src_dst_per_sn_edges[(src, dst, u, v)] /= spr_u_v_normalizer

    __validate_splitting_ratios(net_direct, smart_nodes, flows_src_dst_per_node, active_flows, spr_src_dst_per_sn_edges)
    if logger.level == logging.DEBUG:
        __validate_flows(net_direct, smart_nodes, flows_src_dst_per_node, traffic_matrices_list, demands_ratios, spr_src_dst_per_sn_edges,
                         destination_based_spr)

    logger.info("****Done!****")

    return expected_objective, spr_src_dst_per_sn_edges


def matrices_mcf_LP_with_smart_nodes_solver(smart_nodes, net: NetworkClass, traffic_matrix_list, destination_based_spr):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 1)
    gb_env.setParam(GRB.Param.NumericFocus, Consts.NUMERIC_FOCUS)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.setParam(GRB.Param.Method, Consts.PRIMAL_SIMPLEX)
    gb_env.setParam(GRB.Param.Presolve, Consts.PRESOLVE)
    gb_env.start()

    expected_objective, splitting_ratios_per_src_dst_edge = \
        _aux_mcf_LP_with_smart_nodes_solver(gb_env, net, traffic_matrix_list, destination_based_spr, smart_nodes)
    return smart_nodes, expected_objective, splitting_ratios_per_src_dst_edge
