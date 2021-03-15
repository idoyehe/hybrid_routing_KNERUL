from argparse import ArgumentParser
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_solver
from common.network_class import NetworkClass, nx
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file, error_bound
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
from sys import argv
from random import shuffle
import numpy as np


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


def __initialize_all_weights(net: NetworkClass):
    return np.ones(shape=(net.get_num_edges), dtype=np.float64)


def __stop_loop(net: NetworkClass, current_flows_values, necessary_capacity_dict):
    delta = 0
    for u, v in net.edges:
        edge_idx = net.get_edge2id(u, v)
        delta += np.abs(current_flows_values[edge_idx] - necessary_capacity_dict[0, u, v])
    print('sum[|necessary_capacity_dict - current_flows_values|] = {}'.format(delta))
    return delta < 0.5


def __gradient_decent_update(net: NetworkClass, w_u_v, step_size, current_flows_values, necessary_capacity_dict):
    assert len(necessary_capacity_dict.values()) == len(w_u_v)
    new_w_u_v = np.zeros_like(w_u_v, dtype=np.float64)
    for u, v in net.edges:
        edge_idx = net.get_edge2id(u, v)
        new_w_u_v[edge_idx] = max(0, w_u_v[edge_idx] - step_size * (
                necessary_capacity_dict[0, u, v] - current_flows_values[edge_idx]))
    del w_u_v
    return new_w_u_v


def _main_loop(net, traffic_matrix, necessary_capacity_dict):
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


if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    shuffle(loaded_dict["tms"])
    traffic_matrix = loaded_dict["tms"][0][0]
    opt_value = loaded_dict["tms"][0][1]
    opt_ratio_value, _, r_per_matrix, necessary_capacity_dict, _ = \
        multiple_matrices_mcf_LP_solver(net, [(1.0, traffic_matrix)])
    r_per_matrix = round(r_per_matrix[0], 4)
    assert error_bound(r_per_matrix, opt_value)

    w_u_v, PEFT_congestion = _main_loop(net, traffic_matrix, necessary_capacity_dict)

    print("congestion using link weights is: {}".format(PEFT_congestion))
    print("Congestion using LP: {}".format(r_per_matrix))
    print("Ratio: {}".format(PEFT_congestion / r_per_matrix))
