from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from static_routing.optimal_load_balancing import *
from common.logger import logger
from common.topologies import topology_zoo_loader
import pickle
from common.consts import TMType
import os
from argparse import ArgumentParser
from sys import argv
from pathlib import Path


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from")
    parser.add_argument("-cap", "--default_capacity", type=float, help="The capacity for each edge")
    parser.add_argument("-n", "--total_matrices", type=int, help="The number of total matrices")
    parser.add_argument("-sp", "--sparsity", type=float, help="The sparsity of the matrix", default=0.5)
    parser.add_argument("-m_type", "--tm_type", type=str, help="The type of the matrix")
    parser.add_argument("-e_p", "--elephant_percentage", type=float, help="The percentage of elephant flows")
    parser.add_argument("-n_e", "--network_elephant", type=float, help="The network elephant expectancy", default=400)
    parser.add_argument("-n_m", "--network_mice", type=float, help="The network mice expectancy", default=150)
    options = parser.parse_args(args)
    return options


def _dump_tms_and_opt(net: NetworkClass, default_capacity: float, url: str, matrix_sparsity: float, tm_type,
                      elephant_percentage: float,
                      network_elephant, network_mice, total_matrices: int):
    tms = _generate_traffic_matrix_baseline(net=net,
                                            matrix_sparsity=matrix_sparsity, tm_type=tm_type,
                                            elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                                            network_mice=network_mice,
                                            total_matrices=total_matrices)

    dict2dump = {
        "tms": tms,
        "url": url,
        "capacity": default_capacity,
        "tms_sparsity": matrix_sparsity,
        "tms_type": tm_type, }
    file_name: str = os.getcwd() + "\\..\\TMs_DB\\{}_tms_{}X{}_length_{}_{}_sparsity_{}".format(net.get_name,
                                                                                                net.get_num_nodes,
                                                                                                net.get_num_nodes,
                                                                                                total_matrices, tm_type,
                                                                                                matrix_sparsity)

    from platform import system
    if system() == "Linux":
        file_name = file_name.replace("\\", "/")

    if tm_type == TMType.BIMODAL:
        file_name += "_elephant_percentage_{}".format(elephant_percentage)

    dump_file = open(file_name, 'wb')
    pickle.dump(dict2dump, dump_file)
    dump_file.close()
    return file_name


def _generate_traffic_matrix_baseline(net: NetworkClass, matrix_sparsity: float, tm_type, elephant_percentage: float,
                                      network_elephant, network_mice, total_matrices: int):
    logger.info("Generating baseline of traffic matrices to evaluate of length {}".format(total_matrices))
    tm_list = list()
    for index in range(total_matrices):
        tm = one_sample_tm_base(graph=net.get_g_directed,
                                matrix_sparsity=matrix_sparsity,
                                tm_type=tm_type,
                                elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                                network_mice=network_mice)
        opt_ratio, _ = optimal_load_balancing_LP_solver(net, tm)

        tm_list.append((tm, opt_ratio))
        logger.info("Current TM {} with optimal routing {}".format(index, opt_ratio))

    return tm_list


def load_dump_file(file_name: str):
    dumped_file = open(Path(file_name), 'rb')
    dict2load = pickle.load(dumped_file)
    dumped_file.close()
    assert isinstance(dict2load, dict)
    return dict2load


if __name__ == "__main__":
    args = _getOptions()
    net = NetworkClass(topology_zoo_loader(args.topology_url, default_capacity=args.default_capacity))

    filename: str = _dump_tms_and_opt(net=net, default_capacity=args.default_capacity, url=args.topology_url,
                                      matrix_sparsity=args.sparsity,
                                      tm_type=args.tm_type,
                                      elephant_percentage=args.elephant_percentage,
                                      network_elephant=args.network_elephant,
                                      network_mice=args.network_mice,
                                      total_matrices=args.total_matrices)
    print("Dumps the Tms to:\n{}".format(filename))
