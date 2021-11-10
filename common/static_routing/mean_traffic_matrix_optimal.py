from common.utils import load_dump_file
from common.RL_Envs.optimizer_abstract import Optimizer_Abstract
from common.static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from argparse import ArgumentParser
from sys import argv
import numpy as np


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    args = _getOptions()
    dump_path = args.dumped_path

    loaded_dict = load_dump_file(dump_path)
    topology_gml = loaded_dict["url"]
    net = NetworkClass(topology_zoo_loader(topology_gml))
    traffic_matrix_list = loaded_dict["tms"]
    traffic_matrix_list = np.array([t[0] for t in traffic_matrix_list])
    expected_traffic_matrix = np.mean(traffic_matrix_list, axis=0)
    max_congested_link, necessary_capacity, dst_splitting_ratio = optimal_load_balancing_LP_solver(net, expected_traffic_matrix, return_spr=True)

    traffic_distribution = Optimizer_Abstract(net)
    mean_tm_congestion = np.mean([traffic_distribution._calculating_traffic_distribution(dst_splitting_ratio, t)[0] for t in traffic_matrix_list])
    mean_tm_congestion = np.round(mean_tm_congestion,4)
    print("Mean Traffic Matrix routing scheme congestion: {}".format(mean_tm_congestion))
