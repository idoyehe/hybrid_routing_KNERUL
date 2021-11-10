import sys

sys.path.append("/home/idoye/PycharmProjects/Research_Implementing")

from common.data_generation.tm_generation import one_sample_tm_base
from optimal_load_balancing import optimal_load_balancing_LP_solver
from common.logger import logger
from common.utils import load_dump_file
from common.topologies import topology_zoo_loader
from common.network_class import NetworkClass
from common.consts import TMType, DumpsConsts
from common.RL_Envs.optimizer_abstract import Optimizer_Abstract
from multiple_matrices_MCF import multiple_tms_mcf_LP_solver
from Link_State_Routing_PEFT.gradiant_decent.original_PEFT import PEFT_training_loop_with_init
from argparse import ArgumentParser
from platform import system
from sys import argv
import os
import numpy as np
import pickle


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file", default=None)
    parser.add_argument("-init_w", "--initial_weights", type=eval, help="Calculate Initial Weights", default=False)
    parser.add_argument("-n", "--total_matrices", type=int, help="The number of total matrices", default=20000)
    parser.add_argument("-sp", "--sparsity", type=float, help="The sparsity of the matrix", default=0.3)
    parser.add_argument("-stat_p", "--static_pairs", type=bool, help="Where the pairs with traffic are static",
                        default=False)
    parser.add_argument("-m_type", "--tm_type", type=str, help="The type of the matrix", default=TMType.GRAVITY)
    parser.add_argument("-e_p", "--elephant_percentage", type=float, help="The percentage of elephant flows")
    parser.add_argument("-n_e", "--network_elephant", type=float, help="The network elephant expectancy", default=400)
    parser.add_argument("-n_m", "--network_mice", type=float, help="The network mice expectancy", default=150)
    parser.add_argument("-tail", "--tail_str", type=str, help="String to add in end of the file", default="")
    options = parser.parse_args(args)
    return options


def _get_initial_weights(net, traffic_matrix, necessary_capacity):
    w_u_v = PEFT_training_loop_with_init(net, traffic_matrix, necessary_capacity, 1.0)
    return w_u_v


def calculating_expected_congestion(net_direct, traffic_matrices_list, calc_initial_weights_flag: bool = False):
    expected_objective, _, necessary_capacity_per_tm, optimal_src_dst_splitting_ratios = multiple_tms_mcf_LP_solver(net_direct, traffic_matrices_list)
    aggregate_tm = sum(traffic_matrices_list)
    initial_weights = None
    if calc_initial_weights_flag:
        initial_weights = _get_initial_weights(net_direct, aggregate_tm, necessary_capacity_per_tm[-1])

    traffic_distribution = Optimizer_Abstract(net_direct)
    dst_splitting_ratios = traffic_distribution.reduce_src_dst_spr_to_dst_spr(net_direct,
                                                                              optimal_src_dst_splitting_ratios)
    dst_mean_congestion = np.mean(
        [traffic_distribution._calculating_traffic_distribution(dst_splitting_ratios, t)[0] for t in
         traffic_matrices_list])
    logger.info("Expected Congestion :{}".format(expected_objective))
    logger.info("Dest Mean Congestion Result: {}".format((dst_mean_congestion)))
    return expected_objective, optimal_src_dst_splitting_ratios, initial_weights, dst_mean_congestion


def generate_traffic_matrix_baseline(net: NetworkClass, matrix_sparsity: float, tm_type, static_pairs: bool,
                                     elephant_percentage: float, network_elephant, network_mice, total_matrices: int):
    logger.info("Generating baseline of traffic matrices to evaluate of length {}".format(total_matrices))
    tms_opt_zipped_list = list()
    for index in range(total_matrices):
        tm = one_sample_tm_base(graph=net,
                                matrix_sparsity=matrix_sparsity,
                                static_pairs=static_pairs,
                                tm_type=tm_type,
                                elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                                network_mice=network_mice)

        opt_ratio, _ = optimal_load_balancing_LP_solver(net, tm)

        tms_opt_zipped_list.append((tm, opt_ratio))
        logger.info("Current TM {} with optimal routing {}".format(index, opt_ratio))
    return tms_opt_zipped_list


def dump_dictionary(tail_str, net_direct: NetworkClass, net_path: str, tms_opt_zipped_list, matrix_sparsity: float, tm_type,
                    expected_congestion, optimal_src_dst_splitting_ratios, initial_weights, dst_mean_congestion,
                    static_pairs: bool, elephant_percentage: float, network_elephant, network_mice,
                    total_matrices: int):
    dict2dump = dict()
    dict2dump[DumpsConsts.TMs] = tms_opt_zipped_list
    dict2dump[DumpsConsts.NET_PATH] = net_path
    dict2dump[DumpsConsts.EXPECTED_CONGESTION] = expected_congestion
    dict2dump[DumpsConsts.INITIAL_WEIGHTS] = initial_weights
    dict2dump[DumpsConsts.OPTIMAL_SPLITTING_RATIOS] = optimal_src_dst_splitting_ratios
    dict2dump[DumpsConsts.DEST_EXPECTED_CONGESTION] = dst_mean_congestion
    dict2dump[DumpsConsts.MATRIX_SPARSITY] = matrix_sparsity
    dict2dump[DumpsConsts.MATRIX_TYPE] = tm_type

    folder_name: str = os.getcwd() + "\\..\\TMs_DB\\{}".format(net_direct.get_title)
    file_name: str = os.getcwd() + "\\..\\TMs_DB\\{}\\{}_tms_{}X{}_length_{}_{}_sparsity_{}".format(
        net_direct.get_title,
        net_direct.get_title,
        net_direct.get_num_nodes,
        net_direct.get_num_nodes,
        total_matrices,
        tm_type,
        matrix_sparsity)
    if static_pairs:
        file_name += "_static_pairs"

    if system() == "Linux":
        file_name = file_name.replace("\\", "/")
        folder_name = folder_name.replace("\\", "/")

    if tm_type == TMType.BIMODAL:
        file_name += "_elephant_percentage_{}".format(elephant_percentage)

    file_name += "_{}".format(tail_str)
    os.makedirs(folder_name, exist_ok=True)
    dump_file = open(file_name, 'wb')
    pickle.dump(dict2dump, dump_file)
    dump_file.close()

    return file_name


if __name__ == "__main__":
    args = _getOptions()
    topology_url = args.topology_url
    net_direct = NetworkClass(topology_zoo_loader(topology_url))
    matrix_sparsity = args.sparsity
    tm_type = args.tm_type
    static_pairs = args.static_pairs
    elephant_percentage = args.elephant_percentage
    network_elephant = args.network_elephant
    network_mice = args.network_mice
    total_matrices = args.total_matrices
    dump_path = args.dumped_path
    initial_weights = args.initial_weights
    tail_str = args.tail_str

    if dump_path is None:
        tms_opt_zipped_list = generate_traffic_matrix_baseline(net=net_direct, matrix_sparsity=matrix_sparsity,
                                                               tm_type=tm_type, static_pairs=static_pairs,
                                                               elephant_percentage=elephant_percentage,
                                                               network_elephant=network_elephant,
                                                               network_mice=network_mice, total_matrices=total_matrices)

    else:
        dumps_dict = load_dump_file(dump_path)
        tms_opt_zipped_list = dumps_dict[DumpsConsts.TMs]

    traffic_matrix_list = list(list(zip(*tms_opt_zipped_list))[0])
    expected_objective, optimal_src_dst_splitting_ratios, initial_weights, dst_mean_congestion = calculating_expected_congestion(
        net_direct,
        traffic_matrix_list,
        initial_weights)

    filename: str = dump_dictionary(tail_str=tail_str, net_direct=net_direct, net_path=topology_url,
                                    tms_opt_zipped_list=tms_opt_zipped_list, matrix_sparsity=matrix_sparsity,
                                    tm_type=tm_type,
                                    expected_congestion=expected_objective,
                                    optimal_src_dst_splitting_ratios=optimal_src_dst_splitting_ratios,
                                    initial_weights=initial_weights,
                                    dst_mean_congestion=dst_mean_congestion,
                                    static_pairs=static_pairs,
                                    elephant_percentage=elephant_percentage,
                                    network_elephant=network_elephant,
                                    network_mice=network_mice,
                                    total_matrices=total_matrices)
    print("Dumps the Tms to:\n{}".format(filename))
