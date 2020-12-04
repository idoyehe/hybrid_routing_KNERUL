from Learning_to_Route.common.utils import error_bound
from common.network_class import NetworkClass
from common.consts import EdgeConsts
from common.logger import *
from collections import defaultdict
from static_routing.generating_tms_dumps import load_dump_file
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
                from_its_source = 0
                for out_arches_from_src in net_directed.out_edges_by_node(v):
                    from_its_source += arch_f_vars_dict[out_arches_from_src][flow].x
                assert error_bound(from_its_source, 1.0)
                to_its_src = 0
                for in_arches_to_src in net_directed.in_edges_by_node(v):
                    to_its_src += arch_f_vars_dict[in_arches_to_src][flow].x
                assert error_bound(to_its_src)

            elif v == dst:
                from_its_dst = 0
                for out_arches_from_dst in net_directed.out_edges_by_node(v):
                    from_its_dst += arch_f_vars_dict[out_arches_from_dst][flow].x
                assert error_bound(from_its_dst)

                to_its_dst = 0
                for in_arches_to_dst in net_directed.in_edges_by_node(v):
                    to_its_dst += arch_f_vars_dict[in_arches_to_dst][flow].x
                assert error_bound(to_its_dst, 1.0)
            else:
                assert v not in flow
                to_some_v = 0
                for in_arches_to_v in net_directed.in_edges_by_node(v):
                    to_some_v += arch_f_vars_dict[in_arches_to_v][flow].x
                from_some_v = 0
                for out_arches_from_v in net_directed.out_edges_by_node(v):
                    from_some_v += arch_f_vars_dict[out_arches_from_v][flow].x
                assert error_bound(to_some_v, from_some_v)


def _oblivious_routing(net: NetworkClass, oblivious_ratio=None):
    obliv_lp_problem = gb.Model(name="Applegate's and Cohen's Oblivious Routing LP")
    obliv_lp_problem.setParam(GRB.Param.OutputFlag, 0)
    index = 0

    if oblivious_ratio is None:
        obliv_ratio = obliv_lp_problem.addVar(name="obliv_ratio", lb=0.0, vtype=GRB.CONTINUOUS)
        obliv_lp_problem.setParam(GRB.Attr.ModelSense, GRB.MINIMIZE)
        obliv_lp_problem.setObjectiveN(obliv_ratio, index, 1)
        index += 1
        obliv_lp_problem.update()

    else:
        obliv_ratio = oblivious_ratio

    net_directed = net.get_g_directed

    pi_edges_dict = defaultdict(dict)
    pe_edges_dict = defaultdict(dict)
    f_arch_dict = defaultdict(dict)

    pi_vars_sum = 0
    pe_vars_sum = 0
    f_vars_sum = 0

    for _arch in net_directed.edges:
        _h_sum = 0
        for _h in net_directed.edges:
            pi_edges_dict[_arch][_h] = obliv_lp_problem.addVar(name="PI_{}_({})".format(_arch, _h), lb=0.0,
                                                               vtype=GRB.CONTINUOUS)
            pi_vars_sum += pi_edges_dict[_arch][_h]
            cap_h = net_directed.get_edge_key(_h, EdgeConsts.CAPACITY_STR)
            _h_sum += cap_h * pi_edges_dict[_arch][_h]
        obliv_lp_problem.addConstr(_h_sum, GRB.LESS_EQUAL, obliv_ratio, "SUM(cap(h)*pi_e_(h)<=r;{})".format(_arch))

    obliv_lp_problem.setObjectiveN(pi_vars_sum, index)
    index += 1

    for _arch in net_directed.edges:
        _capacity_arch = net_directed.get_edge_key(_arch, EdgeConsts.CAPACITY_STR)
        for i in range(net_directed.get_num_nodes):
            for j in range(net_directed.get_num_nodes):
                if i == j:
                    pe_edges_dict[_arch][(i, j)] = 0
                    f_arch_dict[_arch][(i, j)] = 0
                else:
                    p_e_i_j = pe_edges_dict[_arch][(i, j)] = \
                        obliv_lp_problem.addVar(name="P_{}_({})".format(_arch, (i, j)), lb=0.0, vtype=GRB.CONTINUOUS)
                    pe_vars_sum += p_e_i_j

                    f_i_j_e = f_arch_dict[_arch][(i, j)] = \
                        obliv_lp_problem.addVar(name="f_{}_({})".format((i, j), _arch), lb=0.0, vtype=GRB.CONTINUOUS)
                    f_vars_sum += f_i_j_e

                    obliv_lp_problem.addConstr(f_i_j_e, GRB.LESS_EQUAL, _capacity_arch * p_e_i_j,
                                               "f_ij({})/cap(e)<=p_{}(i,j)".format(_arch, _arch))

    obliv_lp_problem.update()
    obliv_lp_problem.setObjectiveN(pe_vars_sum, index)
    index += 1
    obliv_lp_problem.setObjectiveN(f_vars_sum, index)
    index += 1

    # flow constrains
    for src, dst in net_directed.get_all_pairs():
        assert src != dst
        flow = (src, dst)

        # Flow conservation at the source
        out_flow_from_source = sum(f_arch_dict[out_arch][flow] for out_arch in net_directed.out_edges_by_node(src))
        in_flow_to_source = sum(f_arch_dict[in_arch][flow] for in_arch in net_directed.in_edges_by_node(src))
        obliv_lp_problem.addConstr(out_flow_from_source - in_flow_to_source, GRB.EQUAL, 1.0,
                                   "{}->{};srcConst".format(src, dst))

        # Flow conservation at the destination
        out_flow_from_dest = sum(f_arch_dict[out_arch][flow] for out_arch in net_directed.out_edges_by_node(dst))
        in_flow_to_dest = sum(f_arch_dict[in_arch][flow] for in_arch in net_directed.in_edges_by_node(dst))
        obliv_lp_problem.addConstr((in_flow_to_dest - out_flow_from_dest), GRB.EQUAL, 1.0,
                                   "{}->{};dstConst".format(src, dst))

        for u in net_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = sum(f_arch_dict[out_arch][flow] for out_arch in net_directed.out_edges_by_node(u))
            in_flow = sum(f_arch_dict[in_arch][flow] for in_arch in net_directed.in_edges_by_node(u))
            obliv_lp_problem.addConstr(out_flow - in_flow, GRB.EQUAL, 0.0, "{}->{};trans_{}_Const".format(src, dst, u))

    obliv_lp_problem.update()

    for _arch_e in net_directed.edges:
        for i in range(net.get_num_nodes):
            for _arc_a in net_directed.edges:
                j = _arc_a[0]
                k = _arc_a[1]
                obliv_lp_problem.addConstr((pi_edges_dict[_arch_e][_arc_a] +
                                            pe_edges_dict[_arch_e][(i, j)] -
                                            pe_edges_dict[_arch_e][(i, k)]), GRB.GREATER_EQUAL, 0.0)

    try:
        logger.info("LP Submit to Solve {}".format(obliv_lp_problem.ModelName))
        obliv_lp_problem.update()
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

    per_arch_flow_fraction = defaultdict(
        lambda: np.zeros(shape=(net_directed.get_num_nodes, net_directed.get_num_nodes), dtype=np.float64))

    per_flow_routing_scheme = np.zeros(shape=(net_directed.get_num_nodes, net_directed.get_num_nodes,
                                              net_directed.get_num_nodes, net_directed.get_num_nodes),
                                       dtype=np.float64)

    for _arch in net_directed.edges:
        for src, dst in net.get_all_pairs():
            assert src != dst
            flow = (src, dst)
            per_arch_flow_fraction[_arch][flow] = f_arch_dict[_arch][flow].x
            per_flow_routing_scheme[flow][_arch] = f_arch_dict[_arch][flow].x

    return obliv_ratio, per_arch_flow_fraction, per_flow_routing_scheme


def _calculate_congestion_per_matrices(net: NetworkClass, traffic_matrix_list: list, oblivious_routing_per_edge: dict):
    logger.info("Calculating congestion to all traffic matrices by {} oblivious routing")
    total_archs_load = np.zeros((net.get_num_nodes, net.get_num_nodes), dtype=np.float64)

    congested_link_histogram = np.zeros((net.get_num_edges,), dtype=np.float64)
    congestion_ratios = list()
    for index, (current_traffic_matrix, current_opt) in enumerate(traffic_matrix_list):
        logger.info("Current matrix index is: {}".format(index))

        assert current_traffic_matrix.shape == (net.get_num_nodes, net.get_num_nodes)

        logger.debug('Calculating the congestion per edge and finding max edge congestion')

        max_congestion = -1
        most_congested_link = None
        for arch, frac_matrix in oblivious_routing_per_edge.items():
            cap_arch = net.get_g_directed.get_edge_key(edge=arch, key=EdgeConsts.CAPACITY_STR)
            link_flow = np.sum(np.multiply(frac_matrix, current_traffic_matrix))

            total_archs_load[arch] += link_flow

            link_congestion = link_flow / cap_arch

            if link_congestion > max_congestion:
                max_congestion = link_congestion
                most_congested_link = arch

        assert max_congestion >= current_opt

        congestion_ratios.append(max_congestion / current_opt)
        congested_link_histogram[net.get_edge2id()[most_congested_link]] += 1

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
        arch = net.get_id2edge()[idx]
        congested_link_fractions.append((arch, congestion))

    congested_link_fractions.sort(key=lambda e: e[1], reverse=True)
    for idx, (arch, congestion) in enumerate(congested_link_fractions):
        print("# {} link {} in most congest in {:.2f}% of the time".format(idx + 1, arch, congestion))


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    save_path = "/".join(dump_path.split("/")[0:-1]) + "/"
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"])).get_g_directed
    oblivious_ratio, oblivious_routing_per_edge, per_flow_routing_scheme = _oblivious_routing(net)
    print("The oblivious ratio for {} is {}".format(net.get_name, oblivious_ratio))
    c_l, total_archs_load, congested_link_histogram = _calculate_congestion_per_matrices(net=net,
                                                                                         traffic_matrix_list=
                                                                                         loaded_dict["tms"],
                                                                                         oblivious_routing_per_edge=oblivious_routing_per_edge)
    print("Average Result: {}".format(np.average(c_l)))
    print("STD Result: {}".format(np.std(c_l)))

    assert np.sum(congested_link_histogram) == len(loaded_dict["tms"])
    congested_link_histogram = 100 * congested_link_histogram / np.sum(congested_link_histogram)

    _sorted_congestion_links(congested_link_histogram)

    oblivious_routing_scheme_name = "{}oblivious_total_load.npy".format(save_path)
    oblivious_routing_scheme_file = open(oblivious_routing_scheme_name, 'wb')
    np.save(oblivious_routing_scheme_file, total_archs_load)
    oblivious_routing_scheme_file.close()
