from static_routing.optimal_load_balancing import *
from static_routing.ecmp_load_balancing import ecmp_arch_congestion
from common.network_class import NetworkClass
from common.consts import EdgeConsts
from static_routing.generating_tms_dumps import load_dump_file
from common.topologies import topology_zoo_loader
from common.logger import logger
from argparse import ArgumentParser
from sys import argv
from random import shuffle


def _calculate_congestion_per_matrices(net: NetworkClass, k: int, traffic_matrix_list: list):
    logger.info("Calculating congestion to all traffic matrices by {} previous average".format(k))

    assert k < len(traffic_matrix_list)
    congestion_ratios = list()
    for index, (current_traffic_matrix, current_opt) in enumerate(traffic_matrix_list[k:]):
        logger.info("Current matrix index: {}\nAveraged matrix calculated based: {}-{}".format(k + index, index, k + index - 1))

        avg_traffic_matrix = np.mean(list(map(lambda t: t[0], traffic_matrix_list[index:index + k])), axis=0)

        assert avg_traffic_matrix.shape == current_traffic_matrix.shape

        logger.debug("Creating LP for averaged matrix")
        # heuristic flows carries
        avg_opt, per_arch_flow_fraction_lp = optimal_load_balancing_LP_solver(net, avg_traffic_matrix)

        logger.debug("Handling the flows that exist in real matrix but not in average one")
        ecmp_flows_matrix = np.zeros(avg_traffic_matrix.shape)
        flows_to_check = np.dstack(np.where((current_traffic_matrix != 0) & (avg_traffic_matrix == 0)))[0]
        for src, dst in flows_to_check:
            assert current_traffic_matrix[src][dst] != 0
            assert avg_traffic_matrix[src][dst] == 0
            ecmp_flows_matrix[src][dst] = float(current_traffic_matrix[src][dst])

        # for flows in average is Zero but in real are non zero
        ecmp_arch_congestion_result = ecmp_arch_congestion(net, ecmp_flows_matrix)

        logger.debug('Calculating the congestion per arch and finding max link congestion')

        max_congested_link = 0
        for u, v, link_capacity in net.get_g_directed.edges.data(EdgeConsts.CAPACITY_STR):
            arch = (u, v)
            fractions_from_lp = per_arch_flow_fraction_lp[arch]
            new_flows_total_congestion_on_link = ecmp_arch_congestion_result[arch]

            heuristic_flows_total_congestion = np.sum(np.multiply(fractions_from_lp, current_traffic_matrix))
            total_link_load = new_flows_total_congestion_on_link + heuristic_flows_total_congestion
            link_congestion = total_link_load / link_capacity
            max_congested_link = max(max_congested_link, link_congestion)

        assert max_congested_link >= current_opt

        congestion_ratios.append(max_congested_link / current_opt)

    return congestion_ratios


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    parser.add_argument("-k", '--average_k', type=int, help="The last average K to based on")
    parser.add_argument("-n", '--number_of_matrices', type=int, help="The number of matrices")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    args = _getOptions()
    dumped_path = args.dumped_path
    k = args.average_k
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    n = args.number_of_matrices
    shuffle(loaded_dict["tms"])
    c_l = _calculate_congestion_per_matrices(net=net, k=k, traffic_matrix_list=loaded_dict["tms"][0:n])
    print(np.average(c_l))
