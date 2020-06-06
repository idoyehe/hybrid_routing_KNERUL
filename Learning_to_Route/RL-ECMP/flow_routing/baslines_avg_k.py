from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass
from consts import EdgeConsts
from generating_tms import load_dump_file
from topologies import topology_zoo_loader
from logger import logger
from argparse import ArgumentParser
from sys import argv
from random import shuffle


def _calculate_congestion_per_matrices(net: NetworkClass, k: int, traffic_matrix_list: list):
    logger.info("Calculating congestion to all traffic matrices by {} previous average".format(k))

    assert k < len(traffic_matrix_list)
    congestion_ratios = list()
    for index, (current_traffic_matrix, current_opt) in enumerate(traffic_matrix_list[k:]):
        logger.info("Current matrix index is: {}".format(k + index))
        logger.info("Average matrix calculated based on: [{},{}]".format(index, k + index - 1))

        avg_traffic_matrix = np.mean(list(map(lambda t: t[0], traffic_matrix_list[index:index + k])), axis=0)

        assert avg_traffic_matrix.shape == current_traffic_matrix.shape

        logger.debug("Solving LP problem for average matrix")
        # heuristic flows carries
        avg_opt, per_edge_flow_fraction_lp = optimal_load_balancing_finder(net, avg_traffic_matrix)

        logger.debug("Handling the flows that exist in real matrix but not in average one")
        ecmp_flows_matrix = np.zeros(avg_traffic_matrix.shape)
        flows_to_check = np.dstack(np.where((current_traffic_matrix != 0) & (avg_traffic_matrix == 0)))[0]
        for src, dst in flows_to_check:
            assert current_traffic_matrix[src][dst] != 0
            assert avg_traffic_matrix[src][dst] == 0
            ecmp_flows_matrix[src][dst] = float(current_traffic_matrix[src][dst])

        # for flows in average is Zero but in real are non zero
        per_edge_flow_fraction_ecmp = get_ecmp_edge_flow_fraction(net, ecmp_flows_matrix)

        logger.debug('Calculating the congestion per edge and finding max link congestion')

        max_congested_link = 0
        for u, v, link_capacity in net.edges.data(EdgeConsts.CAPACITY_STR):
            link = (u, v)
            fractions_from_lp = per_edge_flow_fraction_lp[link]
            new_flows_total_congestion_on_link = per_edge_flow_fraction_ecmp[link]

            heuristic_flows_total_congestion = np.sum(np.multiply(fractions_from_lp, current_traffic_matrix))
            total_link_load = new_flows_total_congestion_on_link + heuristic_flows_total_congestion
            link_congestion = float(total_link_load) / float(link_capacity)

            max_congested_link = max(max_congested_link, link_congestion)

        if not max_congested_link >= current_opt:
            print("BUG!!")
            optimal_load_balancing_finder(net, current_traffic_matrix, max_congested_link)

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
    # shuffle(loaded_dict["tms"])
    c_l = _calculate_congestion_per_matrices(net=net, k=k, traffic_matrix_list=loaded_dict["tms"])
    print(np.average(c_l))
