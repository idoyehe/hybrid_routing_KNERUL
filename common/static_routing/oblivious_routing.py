from common.utils import error_bound, extract_lp_values, load_dump_file
from common.static_routing.multiple_matrices_traffic_distribution import multiple_matrices_traffic_distribution
from common.network_class import NetworkClass
from common.consts import EdgeConsts, Consts, DumpsConsts
from common.logger import *
from common.topologies import topology_zoo_loader
from argparse import ArgumentParser
from sys import argv
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from platform import system
import os, pickle


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-train", "--train_file", type=str, help="The path for the dumped train file")
    parser.add_argument("-test_0", "--test_file_0", type=str, help="The path for the dumped test file")
    parser.add_argument("-test_1", "--test_file_1", type=str, help="The path for the dumped test file")
    parser.add_argument("-test_2", "--test_file_2", type=str, help="The path for the dumped test file")
    parser.add_argument("-lp", "--lp_file", type=str, help="The path for the dumped LP train file")
    parser.add_argument("-save", "--save_dump", type=eval, help="Save the results in ad dump", default=True)
    options = parser.parse_args(args)
    return options


def __validate_solution(net_directed: NetworkClass, arch_f_vars_dict):
    active_flows = [(src, dst) for src, dst in net_directed.get_all_pairs()]

    for flow in active_flows:
        src, dst = flow
        for v in net_directed.nodes:
            if v == src:
                from_its_source = sum(arch_f_vars_dict[flow + out_edge] for out_edge in net_directed.out_edges_by_node(src))
                assert error_bound(from_its_source, 1.0)
                to_its_src = sum(arch_f_vars_dict[flow + in_edge] for in_edge in net_directed.in_edges_by_node(src))
                assert error_bound(to_its_src)

            elif v == dst:
                from_its_dst = sum(arch_f_vars_dict[flow + out_edge] for out_edge in net_directed.out_edges_by_node(dst))
                assert error_bound(from_its_dst)

                to_its_dst = sum(arch_f_vars_dict[flow + in_edge] for in_edge in net_directed.in_edges_by_node(dst))
                assert error_bound(to_its_dst, 1.0)

            else:
                assert v not in flow
                to_some_v = sum(arch_f_vars_dict[flow + in_edge] for in_edge in net_directed.in_edges_by_node(v))
                from_some_v = sum(arch_f_vars_dict[flow + out_edge] for out_edge in net_directed.out_edges_by_node(v))
                assert error_bound(to_some_v, from_some_v)


def oblivious_routing_scheme(net: NetworkClass):
    gb_env = gb.Env(empty=True)
    gb_env.setParam(GRB.Param.OutputFlag, 1)
    gb_env.setParam(GRB.Param.NumericFocus, Consts.NUMERIC_FOCUS)
    gb_env.setParam(GRB.Param.FeasibilityTol, Consts.FEASIBILITY_TOL)
    gb_env.setParam(GRB.Param.Method, Consts.BARRIER_METHOD)
    gb_env.setParam(GRB.Param.Crossover, 0)
    gb_env.setParam(GRB.Param.BarConvTol, Consts.BAR_CONV_TOL)
    gb_env.start()

    oblivious_congestion, src_dst_splitting_ratio = aux_oblivious_routing_scheme(net, gb_env)
    return oblivious_congestion, src_dst_splitting_ratio


def aux_oblivious_routing_scheme(net: NetworkClass, gurobi_env, oblivious_ratio=None) -> object:
    net_directed = net
    oblivious_lp = gb.Model(name="Applegate and Cohen Oblivious Routing LP", env=gurobi_env)

    active_flows = [(src, dst) for src, dst in net_directed.get_all_pairs()]

    if oblivious_ratio is None:
        r_var = oblivious_lp.addVar(name="r", lb=0.0, vtype=GRB.CONTINUOUS)
        oblivious_lp.setObjective(r_var, GRB.MINIMIZE)
    else:
        r_var = oblivious_ratio

    flow_src_dst_edge_vars_dict = oblivious_lp.addVars(active_flows, net_directed.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)

    # f is a routing -> constrains
    for src, dst in active_flows:
        assert src != dst
        flow = (src, dst)
        # Flow conservation at the source
        _flow_out_from_source = sum(flow_src_dst_edge_vars_dict[flow + out_arch] for out_arch in net_directed.out_edges_by_node(src))
        oblivious_lp.addConstrs((flow_src_dst_edge_vars_dict[flow + in_arch] == 0.0 for in_arch in net_directed.in_edges_by_node(src)))
        oblivious_lp.addLConstr(_flow_out_from_source, GRB.EQUAL, 1.0)

        # Flow conservation at the destination
        _flow_out_from_dest = oblivious_lp.addConstrs(
            (flow_src_dst_edge_vars_dict[flow + out_arch] == 0.0 for out_arch in net_directed.out_edges_by_node(dst)))
        _flow_in_to_dest = sum(flow_src_dst_edge_vars_dict[flow + in_arch] for in_arch in net_directed.in_edges_by_node(dst))
        oblivious_lp.addLConstr(_flow_in_to_dest, GRB.EQUAL, 1.0)

        for u in net_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            _flow_out_trans = sum(flow_src_dst_edge_vars_dict[flow + out_arch] for out_arch in net_directed.out_edges_by_node(u))
            _flow_in_trans = sum(flow_src_dst_edge_vars_dict[flow + in_arch] for in_arch in net_directed.in_edges_by_node(u))
            oblivious_lp.addLConstr(_flow_out_trans, GRB.EQUAL, _flow_in_trans)

    oblivious_lp.update()

    pi_edges_vars_dict = oblivious_lp.addVars(net_directed.edges, net_directed.edges, name="PI", lb=0.0, vtype=GRB.CONTINUOUS)

    pe_edges_vars_dict = oblivious_lp.addVars(net_directed.edges, net_directed.nodes, net_directed.nodes, name="Pe", lb=0.0, vtype=GRB.CONTINUOUS)

    for _e in net_directed.edges:
        _e_h_sum = sum(pi_edges_vars_dict[_e + _h] * net_directed.get_edge_key(_h, EdgeConsts.CAPACITY_STR) for _h in net_directed.edges)
        oblivious_lp.addLConstr(_e_h_sum, GRB.LESS_EQUAL, r_var)
        capacity_e = net_directed.get_edge_key(_e, EdgeConsts.CAPACITY_STR)
        oblivious_lp.addConstrs((flow_src_dst_edge_vars_dict[flow + _e] <= pe_edges_vars_dict[_e + flow] * capacity_e for flow in active_flows))

        for i in net_directed.nodes:
            oblivious_lp.addLConstr(pe_edges_vars_dict[_e + (i, i)], GRB.EQUAL, 0.0)
            oblivious_lp.addConstrs(
                (pi_edges_vars_dict[_e + (j, k)] + pe_edges_vars_dict[_e + (i, j)] - pe_edges_vars_dict[_e + (i, k)] >= 0.0 for j, k in
                 net_directed.edges))

    try:
        logger.info("LP Submit to Solve {}".format(oblivious_lp.ModelName))
        oblivious_lp.update()
        oblivious_lp.optimize()
        assert oblivious_lp.Status == GRB.OPTIMAL

    except gb.GurobiError as e:
        raise Exception("Optimize failed due to non-convexity")

    if logger.level == logging.DEBUG:
        oblivious_lp.printStats()
        oblivious_lp.printQuality()

    flow_src_dst_edge_dict = extract_lp_values(flow_src_dst_edge_vars_dict)

    __validate_solution(net_directed, flow_src_dst_edge_dict)

    oblivious_ratio_value = oblivious_ratio
    if oblivious_ratio_value is None:
        oblivious_ratio_value = oblivious_lp.objVal

    src_dst_splitting_ratios = np.zeros(
        shape=(net_directed.get_num_nodes, net_directed.get_num_nodes, net_directed.get_num_nodes, net_directed.get_num_nodes), dtype=np.float64)

    for src, dst in active_flows:
        flow = (src, dst)
        for u in net_directed.nodes:
            if u == dst:
                continue
            flow_out_u = sum(flow_src_dst_edge_dict[flow + out_arch] for out_arch in net_directed.out_edges_by_node(u))
            if flow_out_u > 0.0:
                for _, v in net_directed.out_edges_by_node(u):
                    src_dst_splitting_ratios[src, dst, u, v] = flow_src_dst_edge_dict[(src, dst, u, v)] / flow_out_u
                assert error_bound(sum(src_dst_splitting_ratios[(src, dst)][u]), 1.0)

    return oblivious_ratio_value, src_dst_splitting_ratios


if __name__ == "__main__":
    args = _getOptions()
    train_file = args.train_file
    test_file_0 = args.test_file_0
    test_file_1 = args.test_file_1
    test_file_2 = args.test_file_2
    lp_file = args.lp_file
    save_dump = args.save_dump
    train_loaded_dict = load_dump_file(train_file)
    test_loaded_dict_0 = load_dump_file(test_file_0)
    test_loaded_dict_1 = load_dump_file(test_file_1)
    test_loaded_dict_2 = load_dump_file(test_file_2)
    lp_loaded_dict = load_dump_file(lp_file)

    topology_gml = train_loaded_dict[DumpsConsts.NET_PATH]

    net = NetworkClass(topology_zoo_loader(topology_gml))
    oblivious_ratio, src_dst_splitting_ratios = oblivious_routing_scheme(net)
    print("The oblivious ratio for {} is {}".format(net.get_title, oblivious_ratio))

    train_tms = np.array(list(zip(*train_loaded_dict[DumpsConsts.TMs]))[0])
    train_oblivious_mean_congestion = np.round(multiple_matrices_traffic_distribution(net, train_tms, src_dst_splitting_ratios)[0], 4)
    print("Train Tms: Oblivious Mean Congestion Result: {}".format((train_oblivious_mean_congestion)))

    test_tms_0 = np.array(list(zip(*test_loaded_dict_0[DumpsConsts.TMs]))[0])
    test_oblivious_mean_congestion_0 = np.round(multiple_matrices_traffic_distribution(net, test_tms_0, src_dst_splitting_ratios)[0], 4)
    print("0: Test Tms: Oblivious Mean Congestion Result: {}".format((test_oblivious_mean_congestion_0)))

    test_tms_1 = np.array(list(zip(*test_loaded_dict_1[DumpsConsts.TMs]))[0])
    test_oblivious_mean_congestion_1 = np.round(multiple_matrices_traffic_distribution(net, test_tms_1, src_dst_splitting_ratios)[0], 4)
    print("1: Test Tms: Oblivious Mean Congestion Result: {}".format((test_oblivious_mean_congestion_1)))

    test_tms_2 = np.array(list(zip(*test_loaded_dict_2[DumpsConsts.TMs]))[0])
    test_oblivious_mean_congestion_2 = np.round(multiple_matrices_traffic_distribution(net, test_tms_2, src_dst_splitting_ratios)[0], 4)
    print("2: Test Tms: Oblivious Mean Congestion Result: {}".format((test_oblivious_mean_congestion_2)))

    lp_tms = np.array(list(zip(*lp_file[DumpsConsts.TMs]))[0])
    lp_oblivious_mean_congestion = np.round(multiple_matrices_traffic_distribution(net, lp_tms, src_dst_splitting_ratios)[0], 4)
    print("LP Tms: Oblivious Mean Congestion Result: {}".format((lp_oblivious_mean_congestion)))

    if save_dump:
        dict2dump = dict()
        dict2dump[DumpsConsts.OBLIVIOUS_RATIO] = oblivious_ratio
        dict2dump[DumpsConsts.OBLIVIOUS_MEAN_CONGESTION] = \
            (train_oblivious_mean_congestion,
             test_oblivious_mean_congestion_0,
             test_oblivious_mean_congestion_1,
             test_oblivious_mean_congestion_2,
             lp_oblivious_mean_congestion)
        dict2dump[DumpsConsts.OBLIVIOUS_SRC_DST_SPR] = src_dst_splitting_ratios

        folder_name: str = os.getcwd() + "\\..\\TMs_DB\\{}".format(net.get_title)
        file_name: str = os.getcwd() + "\\..\\TMs_DB\\{}\\{}_oblivious_result".format(net.get_title, net.get_title)

        if system() == "Linux" or system() == "Darwin":
            file_name = file_name.replace("\\", "/")
            folder_name = folder_name.replace("\\", "/")

        os.makedirs(folder_name, exist_ok=True)
        dump_file = open(file_name, 'wb')
        pickle.dump(dict2dump, dump_file)
        dump_file.close()
