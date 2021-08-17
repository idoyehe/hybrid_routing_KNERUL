from common.data_generation.tm_generation import one_sample_tm_base
from optimal_load_balancing import optimal_load_balancing_LP_solver
from oblivious_routing import *
from common.logger import logger
from common.topologies import topology_zoo_loader
from common.consts import TMType
from common.size_consts import SizeConsts
from common.utils import change_zero_cells
import os
from argparse import ArgumentParser
from sys import argv
import pickle


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from")
    parser.add_argument("-obliv", "--oblivious", type=eval, help="Run Oblivious as baseline", default=False)
    parser.add_argument("-n", "--total_matrices", type=int, help="The number of total matrices", default=20000)
    parser.add_argument("-sp", "--sparsity", type=float, help="The sparsity of the matrix", default=0.3)
    parser.add_argument("-stat_p", "--static_pairs", type=bool, help="Where the pairs with traffic are static",
                        default=False)
    parser.add_argument("-m_type", "--tm_type", type=str, help="The type of the matrix", default=TMType.GRAVITY)
    parser.add_argument("-e_p", "--elephant_percentage", type=float, help="The percentage of elephant flows")
    parser.add_argument("-n_e", "--network_elephant", type=float, help="The network elephant expectancy", default=400)
    parser.add_argument("-n_m", "--network_mice", type=float, help="The network mice expectancy", default=150)
    options = parser.parse_args(args)
    return options


def _dump_tms_and_opt(net: NetworkClass,url, matrix_sparsity: float, tm_type,
                      oblivious_routing_per_edge, oblivious_routing_per_flow,
                      static_pairs: bool, elephant_percentage: float, network_elephant, network_mice,
                      total_matrices: int):
    tms = _generate_traffic_matrix_baseline(net=net,
                                            matrix_sparsity=matrix_sparsity, tm_type=tm_type,
                                            oblivious_routing_per_edge=oblivious_routing_per_edge,
                                            static_pairs=static_pairs, elephant_percentage=elephant_percentage,
                                            network_elephant=network_elephant,
                                            network_mice=network_mice,
                                            total_matrices=total_matrices)

    dict2dump = {
        "tms": tms,
        "url": url,
        "oblivious_routing": {
            "per_edge": oblivious_routing_per_edge,
            "per_flow": oblivious_routing_per_flow
        },
        "tms_sparsity": matrix_sparsity,
        "tms_type": tm_type, }

    folder_name: str = os.getcwd() + "\\..\\TMs_DB\\{}".format(net.get_title)
    file_name: str = os.getcwd() + "\\..\\TMs_DB\\{}\\{}_tms_{}X{}_length_{}_{}_sparsity_{}".format(net.get_title,
                                                                                                    net.get_title,
                                                                                                    net.get_num_nodes,
                                                                                                    net.get_num_nodes,
                                                                                                    total_matrices,
                                                                                                    tm_type,
                                                                                                    matrix_sparsity)
    if static_pairs:
        file_name += "_static_pairs"

    from platform import system
    if system() == "Linux":
        file_name = file_name.replace("\\", "/")
        folder_name = folder_name.replace("\\", "/")

    if tm_type == TMType.BIMODAL:
        file_name += "_elephant_percentage_{}".format(elephant_percentage)

    os.makedirs(folder_name, exist_ok=True)
    dump_file = open(file_name, 'wb')
    pickle.dump(dict2dump, dump_file)
    dump_file.close()
    return file_name


def _generate_traffic_matrix_baseline(net: NetworkClass, matrix_sparsity: float, tm_type, oblivious_routing_per_edge,
                                      static_pairs: bool,
                                      elephant_percentage: float,
                                      network_elephant, network_mice, total_matrices: int):
    logger.info("Generating baseline of traffic matrices to evaluate of length {}".format(total_matrices))
    tm_list = list()
    for index in range(total_matrices):
        tm = one_sample_tm_base(graph=net,
                                matrix_sparsity=matrix_sparsity,
                                static_pairs=static_pairs,
                                tm_type=tm_type,
                                elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                                network_mice=network_mice)

        opt_ratio, _ = optimal_load_balancing_LP_solver(net, tm)
        obliv_ratio = None
        if oblivious_routing_per_edge is not None:
            obliv_ratio, _, _ = calculate_congestion_per_matrices(net=net, traffic_matrix_list=[(tm, opt_ratio)],
                                                                  oblivious_routing_per_edge=oblivious_routing_per_edge)
            obliv_ratio = obliv_ratio[0]

        tm_list.append((tm, opt_ratio, obliv_ratio))
        logger.info("Current TM {} with optimal routing {}".format(index, opt_ratio))
    return tm_list


if __name__ == "__main__":
    args = _getOptions()
    net = NetworkClass(topology_zoo_loader(args.topology_url))

    oblivious_routing_per_edge = None
    oblivious_routing_per_flow = None

    if args.oblivious:
        oblivious_ratio, oblivious_routing_per_edge, oblivious_routing_per_flow = oblivious_routing(net)
        print("The oblivious ratio for {} is {}".format(net.get_title, oblivious_ratio))

    filename: str = _dump_tms_and_opt(net=net, url=args.topology_url,
                                      matrix_sparsity=args.sparsity,
                                      tm_type=args.tm_type,
                                      oblivious_routing_per_edge=oblivious_routing_per_edge,
                                      oblivious_routing_per_flow=oblivious_routing_per_flow,
                                      static_pairs=args.static_pairs,
                                      elephant_percentage=args.elephant_percentage,
                                      network_elephant=args.network_elephant,
                                      network_mice=args.network_mice,
                                      total_matrices=args.total_matrices)
    print("Dumps the Tms to:\n{}".format(filename))
