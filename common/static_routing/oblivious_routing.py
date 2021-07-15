from common.utils import error_bound, load_dump_file
from common.network_class import NetworkClass
from common.consts import EdgeConsts, Consts
from common.logger import *
from common.topologies import topology_zoo_loader
from argparse import ArgumentParser
from sys import argv
import numpy as np
import gurobipy as gb
from gurobipy import GRB


def __validate_solution(net_directed: NetworkClass, arch_f_vars_dict):
    assert net_directed.g_is_directed

    for src, dst in net_directed.get_all_pairs():
        flow = (src, dst)
        for v in net_directed.nodes:
            if v == src:
                from_its_source = sum(arch_f_vars_dict[flow + out_arches_from_src].x for out_arches_from_src in
                                      net_directed.out_edges_by_node(v))
                assert error_bound(from_its_source, 1.0)
                to_its_src = sum(arch_f_vars_dict[flow + in_arches_to_src].x for in_arches_to_src in
                                 net_directed.in_edges_by_node(v))
                assert error_bound(to_its_src)

            elif v == dst:
                from_its_dst = sum(arch_f_vars_dict[flow + out_arches_from_dst].x for out_arches_from_dst in
                                   net_directed.out_edges_by_node(v))
                assert error_bound(from_its_dst)

                to_its_dst = sum(arch_f_vars_dict[flow + in_arches_to_dst].x for in_arches_to_dst in
                                 net_directed.in_edges_by_node(v))
                assert error_bound(to_its_dst, 1.0)

            else:
                assert v not in flow
                to_some_v = sum(
                    arch_f_vars_dict[flow + in_arches_to_v].x for in_arches_to_v in net_directed.in_edges_by_node(v))

                from_some_v = sum(arch_f_vars_dict[flow + out_arches_from_v].x for out_arches_from_v in
                                  net_directed.out_edges_by_node(v))
                assert error_bound(to_some_v, from_some_v)


def oblivious_routing(net: NetworkClass):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 0)
    gb_env.setParam(GRB.Param.NumericFocus, 2)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.start()
    prev_obliv_ratio, prev_per_arch_flow_fraction, prev_per_flow_routing_scheme = aux_oblivious_routing(net, gb_env)
    while True:
        try:
            next_obliv_ratio = prev_obliv_ratio - 0.001
            prev_obliv_ratio, prev_per_arch_flow_fraction, prev_per_flow_routing_scheme = aux_oblivious_routing(net,
                                                                                                                gb_env,
                                                                                                                next_obliv_ratio)
            prev_obliv_ratio = next_obliv_ratio
            print("****** Gurobi Failure ******")
        except Exception as e:
            return prev_obliv_ratio, prev_per_arch_flow_fraction, prev_per_flow_routing_scheme


def aux_oblivious_routing(net: NetworkClass, gurobi_env, oblivious_ratio=None):
    obliv_lp_problem = gb.Model(name="Applegate's and Cohen's Oblivious Routing LP", env=gurobi_env)
    obliv_lp_problem.setParam(GRB.Param.OutputFlag, 0)
    objective_index = 0

    if oblivious_ratio is None:
        obliv_ratio = obliv_lp_problem.addVar(name="obliv_ratio", lb=0.0, vtype=GRB.CONTINUOUS)
        obliv_lp_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        obliv_lp_problem.setObjectiveN(obliv_ratio, objective_index, 1)
        objective_index += 1
        obliv_lp_problem.update()

    else:
        obliv_ratio = oblivious_ratio

    net_directed = net

    pi_edges_dict = obliv_lp_problem.addVars(net_directed.edges, net_directed.edges, name="PI", lb=0.0,
                                             vtype=GRB.CONTINUOUS)
    obliv_lp_problem.update()

    for _e in net_directed.edges:
        _e_h_sum = 0
        for _h in net_directed.edges:
            pi_e_h = pi_edges_dict[_e + _h]
            cap_h = net_directed.get_edge_key(_h, EdgeConsts.CAPACITY_STR)
            _e_h_sum += cap_h * pi_e_h
        obliv_lp_problem.addLConstr(_e_h_sum, GRB.LESS_EQUAL, obliv_ratio, "SUM(cap(h)*pi_e_(h)<=r;{})".format(_e))

    obliv_lp_problem.setObjectiveN(sum(dict(pi_edges_dict).values()), objective_index)
    objective_index += 1

    pe_edges_dict = obliv_lp_problem.addVars(net_directed.edges, net_directed.get_num_nodes, net_directed.get_num_nodes,
                                             name="Pe", lb=0.0,
                                             vtype=GRB.CONTINUOUS)
    f_arch_dict = obliv_lp_problem.addVars(net_directed.get_num_nodes, net_directed.get_num_nodes, net_directed.edges,
                                           name="f", lb=0.0, ub=1.0,
                                           vtype=GRB.CONTINUOUS)

    obliv_lp_problem.update()

    for _e in net_directed.edges:
        _capacity_arch = net_directed.get_edge_key(_e, EdgeConsts.CAPACITY_STR)
        for i in net_directed.nodes:
            for j in net_directed.nodes:
                f_i_j_e = f_arch_dict[(i, j) + _e]
                p_e_i_j = pe_edges_dict[_e + (i, j)]
                if i == j:
                    pe_edges_dict[_e + (i, j)] = 0
                    f_arch_dict[(i, j) + _e] = 0
                else:
                    obliv_lp_problem.addLConstr(f_i_j_e, GRB.LESS_EQUAL, _capacity_arch * p_e_i_j,
                                                "f_ij({})/cap(e)<=p_{}(i,j)".format(_e, _e))

    obliv_lp_problem.update()
    obliv_lp_problem.setObjectiveN(sum(dict(pe_edges_dict).values()), objective_index)
    objective_index += 1
    obliv_lp_problem.setObjectiveN(sum(dict(f_arch_dict).values()), objective_index)
    objective_index += 1
    obliv_lp_problem.update()

    for _e in net_directed.edges:
        for i in net_directed.nodes:
            for j, k in net_directed.edges:
                obliv_lp_problem.addLConstr((pi_edges_dict[_e + (j, k)] +
                                             pe_edges_dict[_e + (i, j)] - pe_edges_dict[_e + (i, k)]), GRB.GREATER_EQUAL, 0.0)

    # flow constrains
    for src, dst in net_directed.get_all_pairs():
        assert src != dst
        flow = (src, dst)

        # Flow conservation at the source
        out_flow_from_source = sum(f_arch_dict[flow + out_arch] for out_arch in net_directed.out_edges_by_node(src))
        in_flow_to_source = sum(f_arch_dict[flow + in_arch] for in_arch in net_directed.in_edges_by_node(src))
        obliv_lp_problem.addConstr(out_flow_from_source, GRB.EQUAL, 1.0)
        obliv_lp_problem.addConstr(in_flow_to_source, GRB.EQUAL, 0.0)

        # Flow conservation at the destination
        out_flow_from_dest = sum(f_arch_dict[flow + out_arch] for out_arch in net_directed.out_edges_by_node(dst))
        in_flow_to_dest = sum(f_arch_dict[flow + in_arch] for in_arch in net_directed.in_edges_by_node(dst))
        obliv_lp_problem.addConstr(in_flow_to_dest, GRB.EQUAL, 1.0)
        obliv_lp_problem.addConstr(out_flow_from_dest, GRB.EQUAL, 0.0)

        for u in net_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = sum(f_arch_dict[flow + out_arch] for out_arch in net_directed.out_edges_by_node(u))
            in_flow = sum(f_arch_dict[flow + in_arch] for in_arch in net_directed.in_edges_by_node(u))
            obliv_lp_problem.addConstr(out_flow, GRB.EQUAL, in_flow)

    obliv_lp_problem.update()

    try:
        logger.info("LP Submit to Solve {}".format(obliv_lp_problem.ModelName))
        obliv_lp_problem.update()
        obliv_lp_problem.write("oblivious_lp.mps")
        obliv_lp_problem.optimize()
        assert obliv_lp_problem.Status == GRB.OPTIMAL

    except gb.GurobiError as e:
        raise Exception("Optimize failed due to non-convexity")

    if logger.level == logging.DEBUG:
        obliv_lp_problem.printStats()
        obliv_lp_problem.printQuality()

    __validate_solution(net_directed, f_arch_dict)

    if oblivious_ratio is None:
        obliv_ratio = obliv_lp_problem.objVal

    per_arch_flow_fraction = np.zeros(shape=(net_directed.get_num_nodes, net_directed.get_num_nodes,
                                             net_directed.get_num_nodes, net_directed.get_num_nodes),
                                      dtype=np.float64)

    per_flow_routing_scheme = np.zeros(shape=(net_directed.get_num_nodes, net_directed.get_num_nodes,
                                              net_directed.get_num_nodes, net_directed.get_num_nodes),
                                       dtype=np.float64)

    for _arch in net_directed.edges:
        for src, dst in net.get_all_pairs():
            assert src != dst
            flow = (src, dst)
            per_flow_routing_scheme[flow][_arch] = per_arch_flow_fraction[_arch][flow] = f_arch_dict[flow + _arch].x

    return obliv_ratio, per_arch_flow_fraction, per_flow_routing_scheme


def calculate_congestion_per_matrices(net: NetworkClass, traffic_matrix_list: list, oblivious_routing_per_edge: dict):
    logger.info("Calculating congestion to all traffic matrices by {} oblivious routing")
    total_archs_load = np.zeros((net.get_num_nodes, net.get_num_nodes), dtype=np.float64)

    congested_link_histogram = np.zeros((net.get_num_edges,), dtype=np.float64)
    congestion_ratios = list()
    for index, tple in enumerate(traffic_matrix_list):
        logger.debug("Current matrix index is: {}".format(index))
        current_traffic_matrix = tple[0]
        current_opt = tple[1]

        assert current_traffic_matrix.shape == (net.get_num_nodes, net.get_num_nodes)

        logger.info('Calculating the congestion per edge and finding max edge congestion')

        max_congestion = -1
        most_congested_link = None
        for arch in net.edges:
            frac_matrix = oblivious_routing_per_edge[arch]
            cap_arch = net.get_edge_key(edge=arch, key=EdgeConsts.CAPACITY_STR)
            link_flow = np.sum(np.multiply(frac_matrix, current_traffic_matrix))

            total_archs_load[arch] += link_flow

            link_congestion = link_flow / cap_arch

            if link_congestion > max_congestion:
                max_congestion = link_congestion
                most_congested_link = arch

        assert round(max_congestion, 4) >= current_opt

        congestion_ratios.append(round(max_congestion / current_opt, 4))
        congested_link_histogram[net.get_edge2id(*most_congested_link)] += 1

    return congestion_ratios, total_archs_load, congested_link_histogram


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


def _sorted_congestion_links(congested_link_histogram):
    print("Sorted congestion links")
    congested_link_fractions = list()
    for idx, congestion in enumerate(congested_link_histogram):
        arch = net.get_id2edge(idx)
        congested_link_fractions.append((arch, congestion))

    congested_link_fractions.sort(key=lambda e: e[1], reverse=True)
    for idx, (arch, congestion) in enumerate(congested_link_fractions):
        print("# {} link {} in most congest in {:.2f}% of the time".format(idx + 1, arch, congestion))


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))
    oblivious_ratio, oblivious_routing_per_edge, per_flow_routing_scheme = oblivious_routing(net)
    print("The oblivious ratio for {} is {}".format(net.get_title, oblivious_ratio))

    traffic_matrix_list = loaded_dict["tms"]

    oblivious_values_list = list()

    c_l, total_archs_load, congested_link_histogram = calculate_congestion_per_matrices(net=net,
                                                                                        traffic_matrix_list=loaded_dict["tms"],
                                                                                        oblivious_routing_per_edge=oblivious_routing_per_edge)
    print("Average Result: {}".format(np.average(c_l)))
    print("STD Result: {}".format(np.std(c_l)))

    assert np.sum(congested_link_histogram) == len(loaded_dict["tms"])
    congested_link_histogram = 100 * congested_link_histogram / np.sum(congested_link_histogram)

    _sorted_congestion_links(congested_link_histogram)
