from argparse import ArgumentParser
from common.static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
from sys import argv
import numpy as np
from tabulate import tabulate


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


def __initialize_all_weights(net: NetworkClass):
    return np.ones(shape=(net.get_num_edges), dtype=np.float64) * 10


def __stop_loop(net: NetworkClass, current_flows_values, necessary_capacity):
    delta = 0
    for u, v in net.edges:
        edge_idx = net.get_edge2id(u, v)
        delta += np.abs(current_flows_values[edge_idx] - necessary_capacity[edge_idx])
    print('sum[|necessary_capacity_dict - current_flows_values|] = {}'.format(delta))
    return delta <0.5


def __gradient_decent_update(net: NetworkClass, w_u_v, step_size, current_flows_values, necessary_capacity):
    assert len(necessary_capacity) == len(w_u_v)
    new_w_u_v = np.zeros_like(w_u_v, dtype=np.float64)
    for u, v in net.edges:
        edge_idx = net.get_edge2id(u, v)
        new_w_u_v[edge_idx] = max(0, w_u_v[edge_idx] - step_size * (
                necessary_capacity[edge_idx] - current_flows_values[edge_idx]))
    del w_u_v
    return new_w_u_v


def PEFT_main_loop(net, traffic_matrix, necessary_capacity):
    step_size = 1 / max(necessary_capacity)
    traffic_distribution = PEFTOptimizer(net, None)
    w_u_v = __initialize_all_weights(net)
    max_congestion, most_congested_link, total_congestion, total_congestion_per_link, current_flows_values = \
        traffic_distribution.step(w_u_v, traffic_matrix, None)

    while __stop_loop(net, current_flows_values, necessary_capacity) == False:
        w_u_v = __gradient_decent_update(net, w_u_v, step_size, current_flows_values, necessary_capacity)
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
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))

    traffic_matrix = loaded_dict["tms"][0][0]

    heuristic_optimal,necessary_capacity = optimal_load_balancing_LP_solver(net,traffic_matrix)

    w_u_v, PEFT_congestion = PEFT_main_loop(net, traffic_matrix, necessary_capacity)

    print(w_u_v)
