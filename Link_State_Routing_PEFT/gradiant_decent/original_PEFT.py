from argparse import ArgumentParser
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_baseline_solver, \
    multiple_matrices_mcf_LP_heuristic_solver
from common.network_class import NetworkClass, nx
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file, error_bound
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


def __initialize_all_weights(net: NetworkClass):
    return np.ones(shape=(net.get_num_edges), dtype=np.float64) * 10


def __stop_loop(net: NetworkClass, current_flows_values, necessary_capacity_dict):
    delta = 0
    for u, v in net.edges:
        edge_idx = net.get_edge2id(u, v)
        delta += np.abs(current_flows_values[edge_idx] - necessary_capacity_dict[u, v])
    print('sum[|necessary_capacity_dict - current_flows_values|] = {}'.format(delta))
    return delta <0.5


def __gradient_decent_update(net: NetworkClass, w_u_v, step_size, current_flows_values, necessary_capacity_dict):
    assert len(necessary_capacity_dict.values()) == len(w_u_v)
    new_w_u_v = np.zeros_like(w_u_v, dtype=np.float64)
    for u, v in net.edges:
        edge_idx = net.get_edge2id(u, v)
        new_w_u_v[edge_idx] = max(0, w_u_v[edge_idx] - step_size * (
                necessary_capacity_dict[u, v] - current_flows_values[edge_idx]))
    del w_u_v
    return new_w_u_v


def PEFT_main_loop(net, traffic_matrix, necessary_capacity_dict):
    step_size = 1 / max(necessary_capacity_dict.values())
    traffic_distribution = PEFTOptimizer(net, None)
    w_u_v = __initialize_all_weights(net)
    max_congestion, most_congested_link, total_congestion, total_congestion_per_link, current_flows_values = \
        traffic_distribution.step(w_u_v, traffic_matrix, None)

    while __stop_loop(net, current_flows_values, necessary_capacity_dict) == False:
        w_u_v = __gradient_decent_update(net, w_u_v, step_size, current_flows_values, necessary_capacity_dict)
        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, current_flows_values = \
            traffic_distribution.step(w_u_v, traffic_matrix, None)

    max_congestion, most_congested_link, total_congestion, total_congestion_per_link, current_flows_values = \
        traffic_distribution.step(w_u_v, traffic_matrix, None)
    return w_u_v, max_congestion


def experiment(traffic_matrix_list, w_u_v, baseline_objective, r_per_mtrx):
    traffic_distribution = PEFTOptimizer(net, None)
    heursitic_value = 0

    data = list()
    headers = ["# Matrix",
               "Congestion heuristic link weights",
               "Congestion using multiple MCF - LP",
               "Congestion using LP optimal",
               "Congestion Ratio, link weights / multiple MCF"]
    for idx, (pr, tm, opt) in enumerate(traffic_matrix_list):
        max_congestion, _, _, _, _ = traffic_distribution.step(w_u_v, tm, None)
        heursitic_value += pr * max_congestion
        data.append([idx, max_congestion, r_per_mtrx[idx], opt, max_congestion / r_per_mtrx[idx]])

    print(tabulate(data, headers=headers))
    print("Heuristic objective Vs. expected objective baseline: {}".format(heursitic_value / baseline_objective))
    return heursitic_value


if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    shuffle(loaded_dict["tms"])

    l = 2
    p = [0.99] + [(1 - 0.99) / (l - 1)] * (l - 1)
    traffic_matrix_list = [(p[i], t[0]) for i, t in enumerate(loaded_dict["tms"][0:l])]

    baseline_objective, _, r_per_mtrx, _ = multiple_matrices_mcf_LP_baseline_solver(net, traffic_matrix_list)
    heuristic_optimal, splitting_ratios_per_src_dst_edge, necessary_capacity_dict = multiple_matrices_mcf_LP_heuristic_solver(
        net, traffic_matrix_list)

    expected_tm = sum(pr * t for pr, t in traffic_matrix_list)

    w_u_v, PEFT_congestion = PEFT_main_loop(net, expected_tm, necessary_capacity_dict)

    traffic_matrix_list = [(p[i], t[0], t[1]) for i, t in enumerate(loaded_dict["tms"][0:l])]

    experiment(traffic_matrix_list, w_u_v, baseline_objective, r_per_mtrx)
