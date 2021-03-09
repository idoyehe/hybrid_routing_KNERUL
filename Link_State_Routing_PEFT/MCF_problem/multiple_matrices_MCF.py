from common.consts import EdgeConsts
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.logger import *
from sys import argv
import gurobipy as gb
from gurobipy import GRB
from argparse import ArgumentParser
from common.utils import load_dump_file, error_bound, extract_flows

R = 10


def __validate_splitting_ratios(net_direct, flows, splitting_ratios_per_src_dst_edge):
    for u in net_direct.nodes():
        if len(net_direct.out_edges_by_node(u)) == 0:
            continue
        for src, dst in flows:
            splitting_ratios_sum = 0.0
            for _, v in net_direct.out_edges_by_node(u):
                s_r_t_out_arch = splitting_ratios_per_src_dst_edge[src, dst, u, v]
                assert s_r_t_out_arch >= 0.0
                splitting_ratios_sum += s_r_t_out_arch
            assert error_bound(splitting_ratios_sum, 1.0)


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

                for _, v in net_direct.out_edges_by_node(u):
                    assert error_bound(flows_per_edge_src_dst[src, dst, u, v],
                                       from_dst * splitting_ratios_per_src_dst_edge[src, dst, u, v])


            else:
                assert u not in (src, dst)
                to_transit_u = sum(flows_per_edge_src_dst[src, dst, v, u] for v, _ in net_direct.in_edges_by_node(u))
                from_transit_u = sum(flows_per_edge_src_dst[src, dst, u, v] for _, v in net_direct.out_edges_by_node(u))
                assert error_bound(to_transit_u, from_transit_u)

                for _, v in net_direct.out_edges_by_node(u):
                    assert error_bound(flows_per_edge_src_dst[src, dst, u, v],
                                       from_transit_u * splitting_ratios_per_src_dst_edge[
                                           src, dst, u, v])

    for key, value in flows_per_edge_src_dst.items():
        if (key[0], key[1]) not in current_flows:
            assert value == 0.0


def __validate_flow(net_direct, traffic_matrix_list, flows_per_mtrx_src_dst_per_edge,
                    splitting_ratios_edge_src_dst):
    for m_index, (_, tm) in enumerate(traffic_matrix_list):
        current_matrix_flows_vars_per_dest = dict()
        for key, val in flows_per_mtrx_src_dst_per_edge.items():
            if key[0] == m_index:
                current_matrix_flows_vars_per_dest[key[1:]] = val

        __validate_flow_per_matrix(net_direct, tm, current_matrix_flows_vars_per_dest, splitting_ratios_edge_src_dst)


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


def _aux_mcf_LP_solver(net: NetworkClass, traffic_matrices_list, gurobi_env, opt_ratio_value=None):
    """Preparation"""
    mcf_problem = gb.Model(name="MCF problem for mean MCF, given network, TM list and probabilities",
                           env=gurobi_env)

    traffic_matrices_list_length = len(traffic_matrices_list)

    total_demands = sum(t for _, t in traffic_matrices_list)

    flows = extract_flows(total_demands)

    net_direct = net
    del net

    vars_flows_src_dst_per_edge = mcf_problem.addVars(flows, net_direct.edges,
                                                      name="f", lb=0.0, vtype=GRB.CONTINUOUS)

    vars_flows_per_mtrx_src_dst_per_edge = mcf_problem.addVars(traffic_matrices_list_length,
                                                               flows, net_direct.edges,
                                                               name="f_m", lb=0.0, vtype=GRB.CONTINUOUS)

    vars_r_per_mtrx = mcf_problem.addVars(traffic_matrices_list_length, name="r", lb=0.0, vtype=GRB.CONTINUOUS)

    mcf_problem.update()
    """Building Constraints"""
    total_objective = sum(tm_prb * vars_r_per_mtrx[m_idx] for m_idx, (tm_prb, _) in enumerate(traffic_matrices_list))

    if opt_ratio_value is None:
        mcf_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        mcf_problem.setObjective(total_objective)
    else:
        mcf_problem.addConstr(total_objective <= opt_ratio_value)
    mcf_problem.update()

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

    if opt_ratio_value is None:
        opt_ratio_value = round(mcf_problem.objVal, R)

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

    __validate_solution(net_direct, flows, traffic_matrix_list, splitting_ratios_per_src_dst_edge,
                        flows_per_mtrx_src_dst_per_edge)

    necessary_capacity_dict = dict()
    for m_index in range(traffic_matrices_list_length):
        for u, v in net_direct.edges:
            necessary_capacity = 0
            for src, dst in flows:
                necessary_capacity += flows_per_mtrx_src_dst_per_edge[m_index, src, dst, u, v]
            necessary_capacity_dict[m_index, u, v] = necessary_capacity

    src_dst_path_prob = create_paths_probability(net_direct, flows, splitting_ratios_per_src_dst_edge)

    return opt_ratio_value, splitting_ratios_per_src_dst_edge, r_per_mtrx, necessary_capacity_dict, src_dst_path_prob


def multiple_matrices_mcf_LP_solver(net: NetworkClass, traffic_matrix_list):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 3)
    gb_env.setParam(GRB.Param.FeasibilityTol, 1e-9)
    gb_env.start()

    opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict, src_dst_path_prob = _aux_mcf_LP_solver(
        net,
        traffic_matrix_list,
        gb_env)
    while True:
        try:
            opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict, src_dst_path_prob = _aux_mcf_LP_solver(
                net,
                traffic_matrix_list,
                gb_env,
                opt_ratio_value - 0.001)
            print("****** Gurobi Failure ******")
            opt_ratio_value -= 0.001
        except Exception as e:
            return opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict, src_dst_path_prob


def __create_paths_probability_src_dst(net: NetworkClass, source, dest, splitting_ratios_per_src_dst_edge):
    all_simple_path = net.all_simple_paths(source, dest)
    path_prob = dict()
    for p_i in all_simple_path:
        key = str(p_i)
        path_prob[key] = 1.0
        p_i_edges = [(p_i[i], p_i[1:][i]) for i in range(len(p_i[1:]))]
        for u, v in p_i_edges:
            path_prob[key] *= splitting_ratios_per_src_dst_edge[source, dest, u, v]
    # assert sum(path_prob.values()) == 1.0
    return path_prob


def create_paths_probability(net: NetworkClass, flows, splitting_ratios_per_src_dst_edge):
    src_dst_path_prob = dict()
    for src, dst in flows:
        src_dst_path_prob[src, dst] = __create_paths_probability_src_dst(net, src, dst,
                                                                         splitting_ratios_per_src_dst_edge)
    return src_dst_path_prob


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(
        topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    from random import shuffle

    shuffle(loaded_dict["tms"])
    l = 10
    p = [0.99] + [(1 - 0.99) / (l - 1)] * (l - 1)
    traffic_matrix_list = [(p[i], t[0]) for i, t in enumerate(loaded_dict["tms"][0:l])]
    opt_ratio_value, splitting_ratios_per_src_dst_edge, r_vars_per_matrix, necessary_capacity_dict, src_dst_path_prob = \
        multiple_matrices_mcf_LP_solver(net, traffic_matrix_list)

    for i, t_elem in enumerate(loaded_dict["tms"][0:l]):
        assert r_vars_per_matrix[i] >= t_elem[1] or error_bound(r_vars_per_matrix[i], t_elem[1])
    pass
