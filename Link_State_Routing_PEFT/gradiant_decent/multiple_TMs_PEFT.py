from argparse import ArgumentParser
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_baseline_solver
from common.network_class import NetworkClass, nx
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file, extract_flows
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
from sys import argv
from random import shuffle
import numpy as np
from tabulate import tabulate


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


def __initialize_all_lambadas(net: NetworkClass, number_of_matrices):
    return np.random.rand(number_of_matrices, net.get_num_edges)*10


def _run_single_src_dst_demand(net: NetworkClass, demand_src_dst, weights_src_dst, src: int, dst: int):
    traffic_distribution = PEFTOptimizer(net, None)
    tm = np.zeros(shape=(net.get_num_nodes, net.get_num_nodes), dtype=np.float64)
    tm[src, dst] = demand_src_dst
    _, _, _, _, current_flows_values = traffic_distribution.step(weights_src_dst, tm, None)
    return current_flows_values


def _update_weights(traffic_matrices_list, lambadas):
    total_demands = sum(t for t in traffic_matrices_list)
    flows = extract_flows(total_demands)
    weights_per_src_dst = dict()
    for src, dst in flows:
        weights_per_src_dst[(src, dst)] = np.zeros(shape=(net.get_num_edges), dtype=np.float64)
        numerator = 0
        denominator = total_demands[src, dst]
        for u, v in net.edges:
            edge_idx = net.get_edge2id(u, v)
            for idx, tm in enumerate(traffic_matrices_list):
                numerator += tm[src, dst] * lambadas[idx, edge_idx]
            weights_per_src_dst[(src, dst)][edge_idx] = numerator / denominator
    return weights_per_src_dst


def _run_single_tm(net: NetworkClass, tm, weights_per_src_dst):
    flows = extract_flows(tm)
    tm_current_flows_values = np.zeros((net.get_num_edges), dtype=np.float64)
    for src, dst in flows:
        current_weights = weights_per_src_dst[(src, dst)]
        assert len(current_weights) == net.get_num_edges
        tm_current_flows_values += _run_single_src_dst_demand(net, tm[src, dst], current_weights, src, dst)
    return tm_current_flows_values


def _update_lambadas(net: NetworkClass, lambadas, current_tms_flows_values, necessary_capacity_per_matrix_dict, number_of_matrices):
    for idx in range(number_of_matrices):
        tm_necessary_capacity = np.zeros(shape=(net.get_num_edges), dtype=np.float64)
        for u, v in net.edges:
            edge_idx = net.get_edge2id(u, v)
            tm_necessary_capacity[edge_idx] = necessary_capacity_per_matrix_dict[(idx, u, v)]
        current_step_size = 1 / max(tm_necessary_capacity)
        for u, v in net.edges:
            edge_idx = net.get_edge2id(u, v)
            lambadas[idx, edge_idx] = max(0, lambadas[idx, edge_idx] - current_step_size * (
                        tm_necessary_capacity[edge_idx] - current_tms_flows_values[idx][edge_idx]))
    return lambadas


def __stop_loop(net: NetworkClass, current_flows_per_tm, necessary_capacity_dict, number_of_matrices):
    delta = 0
    for idx in range(number_of_matrices):
        for u, v in net.edges:
            edge_idx = net.get_edge2id(u, v)
            delta += np.abs(current_flows_per_tm[idx][edge_idx] - necessary_capacity_dict[(idx, u, v)])
    print('sum[|necessary_capacity_dict - current_flows_values|] = {}'.format(delta))
    return delta < 0.5


def PEFT_main_loop(net, traffic_matrix_list, necessary_capacity_per_matrix_dict):
    number_of_matrices = len(traffic_matrix_list)
    lambadas = __initialize_all_lambadas(net, number_of_matrices)
    weights = _update_weights(traffic_matrix_list, lambadas)
    current_flows_per_tm = dict()
    for idx, tm in enumerate(traffic_matrix_list):
        current_flows_per_tm[idx] = _run_single_tm(net, tm, weights)

    while not __stop_loop(net, current_flows_per_tm, necessary_capacity_per_matrix_dict, number_of_matrices):
        lambadas = _update_lambadas(net, lambadas, current_flows_per_tm, necessary_capacity_per_matrix_dict, number_of_matrices)
        weights = _update_weights(traffic_matrix_list, lambadas)
        for idx, tm in enumerate(traffic_matrix_list):
            current_flows_per_tm[idx] = _run_single_tm(net, tm, weights)

    return weights, current_flows_per_tm


if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    shuffle(loaded_dict["tms"])

    l = 2
    p = [1 / l] * l
    traffic_matrix_list = [(p[i], t[0]) for i, t in enumerate(loaded_dict["tms"][0:l])]

    _, _, _, necessary_capacity_per_matrix_dict = multiple_matrices_mcf_LP_baseline_solver(net, traffic_matrix_list)
    traffic_matrix_list = [t[0] for i, t in enumerate(loaded_dict["tms"][0:l])]

    weights, current_flows_per_tm = PEFT_main_loop(net, traffic_matrix_list, necessary_capacity_per_matrix_dict)
    pass
