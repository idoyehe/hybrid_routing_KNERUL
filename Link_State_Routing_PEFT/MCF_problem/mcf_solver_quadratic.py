from common.consts import EdgeConsts
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.logger import *
import gurobipy as gb
from gurobipy import GRB
from argparse import ArgumentParser
from sys import argv
from common.utils import load_dump_file, error_bound

R = 9


def __validate_splitting_ratios(net_direct, splitting_ratios_vars_per_dest):
    for t in net_direct.nodes:
        for u in net_direct.nodes:
            splitting_ratios_sum = 0.0
            res_sum = 1.0
            if u == t:
                res_sum = 0  # because no flow destined to t get out of t

            for out_arch in net_direct.out_edges_by_node(u):
                s_r_t_out_arch = splitting_ratios_vars_per_dest[(t,) + out_arch]
                assert s_r_t_out_arch >= 0.0
                splitting_ratios_sum += s_r_t_out_arch
            assert error_bound(splitting_ratios_sum, res_sum)


def __validate_flow_per_matrix(net_direct, tm, current_matrix_flows_vars_per_dest, splitting_ratios_vars_per_dest):
    assert len(current_matrix_flows_vars_per_dest) == net_direct.get_num_nodes * net_direct.get_num_edges

    for t in net_direct.nodes:
        for s in net_direct.nodes:
            if t == s:
                for out_arch in net_direct.out_edges_by_node(t):
                    assert error_bound(current_matrix_flows_vars_per_dest[(t,) + out_arch])
                in_going_flows_to_t = 0
                for in_arch in net_direct.in_edges_by_node(t):
                    assert current_matrix_flows_vars_per_dest[(t,) + in_arch] >= 0
                    in_going_flows_to_t += current_matrix_flows_vars_per_dest[(t,) + in_arch]
                assert error_bound(in_going_flows_to_t, sum(tm[:, t]))
                continue

            ingoing_flows = 0
            for in_arch in net_direct.in_edges_by_node(s):
                assert current_matrix_flows_vars_per_dest[(t,) + in_arch] >= 0
                ingoing_flows += current_matrix_flows_vars_per_dest[(t,) + in_arch]

            outgoing_flow = 0
            for out_arch in net_direct.out_edges_by_node(s):
                assert current_matrix_flows_vars_per_dest[(t,) + out_arch] >= 0
                outgoing_flow += current_matrix_flows_vars_per_dest[(t,) + out_arch]
                assert error_bound(current_matrix_flows_vars_per_dest[(t,) + out_arch],
                                   splitting_ratios_vars_per_dest[(t,) + out_arch] * (ingoing_flows + tm[s, t]))

            assert error_bound(outgoing_flow - ingoing_flows, tm[s, t])


def __validate_flow(net_direct, traffic_matrix_list, flows_vars_per_matrix_index_per_dest,
                    splitting_ratios_vars_per_dest):
    for m_index, (_, tm) in enumerate(traffic_matrix_list):
        current_matrix_flows_vars_per_dest = dict()
        for key, val in flows_vars_per_matrix_index_per_dest.items():
            if key[0] == m_index:
                current_matrix_flows_vars_per_dest[key[1:]] = val

        __validate_flow_per_matrix(net_direct, tm, current_matrix_flows_vars_per_dest, splitting_ratios_vars_per_dest)


def __extract_values(gurobi_vars_dict):
    gurobi_vars_dict = dict(gurobi_vars_dict)

    for key in gurobi_vars_dict.keys():
        gurobi_vars_dict[key] = round(gurobi_vars_dict[key].x, R)
    return gurobi_vars_dict


def __validate_solution(net_direct, traffic_matrix_list, splitting_ratios_vars_per_dest,
                        flows_vars_per_matrix_index_per_dest):
    __validate_splitting_ratios(net_direct, splitting_ratios_vars_per_dest)
    __validate_flow(net_direct, traffic_matrix_list, flows_vars_per_matrix_index_per_dest,
                    splitting_ratios_vars_per_dest)


def mcf_QP_solver(net: NetworkClass, traffic_matrix_list):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 3)
    gb_env.setParam(GRB.Param.NonConvex, 2)
    gb_env.setParam(GRB.Param.FeasibilityTol, 1e-9)
    gb_env.start()

    opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict = _aux_mcf_LP_solver(
        net,
        traffic_matrix_list,
        gb_env)
    while True:
        try:
            opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict = _aux_mcf_LP_solver(
                net,
                traffic_matrix_list,
                gb_env,
                opt_ratio_value - 0.001)
            print("****** Gurobi Failure ******")
            opt_ratio_value -= 0.001
        except Exception as e:
            return opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict


def _aux_mcf_LP_solver(net: NetworkClass, tm_list, gurobi_env, opt_ratio_value=None):
    opt_qp_problem = gb.Model(name="QP problem for mean MCF, given network and TM list and probabilities",
                              env=gurobi_env)

    traffic_matrix_list_length = len(tm_list)

    net_direct = net.get_g_directed
    splitting_ratios_vars_per_dest = opt_qp_problem.addVars(net_direct.nodes, net_direct.edges, name="x", lb=0.0,
                                                            ub=1.000000000, vtype=GRB.CONTINUOUS)
    opt_qp_problem.update()

    for t in net_direct.nodes:
        for u in net_direct.nodes:

            sum_res = 1.0
            if u == t:  # target is the u means no splitting ratios, all stay in target means u
                sum_res = 0.0

            ratios_sum = sum(splitting_ratios_vars_per_dest[(t, u, v)] for _, v in net_direct.out_edges_by_node(u))
            opt_qp_problem.addConstr(ratios_sum == sum_res)

    flows_vars_per_matrix_index_per_dest = opt_qp_problem.addVars(traffic_matrix_list_length, net_direct.nodes,
                                                                  net_direct.edges,
                                                                  name="f", lb=0.0, vtype=GRB.CONTINUOUS)
    r_vars_per_matrix = opt_qp_problem.addVars(traffic_matrix_list_length)

    opt_qp_problem.update()

    for m_index in range(0, len(tm_list)):
        for u, v in net_direct.edges:
            arch = (u, v)
            capacity = net_direct.get_edge_key(arch, EdgeConsts.CAPACITY_STR)
            link_utilization = sum(flows_vars_per_matrix_index_per_dest[(m_index, t, u, v)] for t in net_direct.nodes)
            opt_qp_problem.addConstr(link_utilization <= capacity * r_vars_per_matrix[m_index])

        for m_index, (_, tm) in enumerate(tm_list):

            for s in net_direct.nodes:
                for t in net_direct.nodes:
                    if s == t:
                        opt_qp_problem.addConstrs(flows_vars_per_matrix_index_per_dest[(m_index, t, t, v)] == 0
                                                  for _, v in net_direct.out_edges_by_node(t))

                        _collected_flow_in_t_destined_t = sum(flows_vars_per_matrix_index_per_dest[(m_index, t, v, t)]
                                                              for v, _ in net_direct.in_edges_by_node(t))
                        opt_qp_problem.addConstr(_collected_flow_in_t_destined_t == sum(tm[:, t]))
                        continue

                    _collected_flow_in_s_destined_t = sum(flows_vars_per_matrix_index_per_dest[(m_index, t, v, s)]
                                                          for v, _ in net_direct.in_edges_by_node(s)) + tm[s, t]

                    _outgoing_flow_from_s_destined_t = sum(flows_vars_per_matrix_index_per_dest[(m_index, t, s, v)]
                                                           for _, v in net_direct.out_edges_by_node(s))
                    opt_qp_problem.addConstr(_collected_flow_in_s_destined_t == _outgoing_flow_from_s_destined_t)

                    for _, v in net_direct.out_edges_by_node(s):
                        opt_qp_problem.addConstr(
                            flows_vars_per_matrix_index_per_dest[(m_index, t, s, v)] ==
                            _collected_flow_in_s_destined_t * splitting_ratios_vars_per_dest[(t, s, v)])

    total_objective = sum(tm_prob * r_vars_per_matrix[m_index] for m_index, (tm_prob, _) in enumerate(tm_list))

    if opt_ratio_value is None:
        opt_qp_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        opt_qp_problem.setObjective(total_objective)
    else:
        opt_qp_problem.addConstr(total_objective <= opt_ratio_value)
    opt_qp_problem.update()

    try:
        logger.info("LP Submit to Solve {}".format(opt_qp_problem.ModelName))
        opt_qp_problem.update()
        opt_qp_problem.optimize()
        assert opt_qp_problem.Status == GRB.OPTIMAL
    except AssertionError as e:
        raise Exception("****Optimize failed****\nAssertion Error:\n{}".format(e))

    except gb.GurobiError as e:
        raise Exception("****Optimize failed****\nException is:\n{}".format(e))

    if opt_ratio_value is None:
        opt_ratio_value = round(opt_qp_problem.objVal, R)

    if logger.level == logging.DEBUG:
        opt_qp_problem.printStats()
        opt_qp_problem.printQuality()

    splitting_ratios_vars_per_dest = __extract_values(splitting_ratios_vars_per_dest)
    flows_vars_per_matrix_index_per_dest = __extract_values(flows_vars_per_matrix_index_per_dest)
    r_vars_per_matrix = __extract_values(r_vars_per_matrix)

    opt_qp_problem.close()
    __validate_solution(net_direct, tm_list, splitting_ratios_vars_per_dest,
                        flows_vars_per_matrix_index_per_dest)

    necessary_capacity_dict = dict()
    for m_index, (_, _) in enumerate(tm_list):
        for arch in net_direct.edges:
            necessary_capacity = 0
            for t in net_direct.nodes:
                necessary_capacity += flows_vars_per_matrix_index_per_dest[(m_index, t) + arch]
            necessary_capacity_dict[(m_index, arch)] = necessary_capacity

    return opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(
        topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"])).get_g_directed
    from random import shuffle

    l = 3
    traffic_matrix_list = [(1 / l, t[0]) for i, t in enumerate(loaded_dict["tms"][0:l])]
    opt_ratio_value, splitting_ratios_vars_per_dest, r_vars_per_matrix, necessary_capacity_dict = \
        mcf_QP_solver(net, traffic_matrix_list)
    pass

    # assert round(opt_ratio_value, 4) == round(loaded_dict["tms"][0][1], 4)
