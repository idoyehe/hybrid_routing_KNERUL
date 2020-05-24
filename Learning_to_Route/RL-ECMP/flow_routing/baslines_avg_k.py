from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass
from consts import EdgeConsts
from generating_tms import load_dump_file
from topologies import topology_zoo_loader
from logger import logger
from argparse import ArgumentParser
from sys import argv


def _calculate_congestion_per_matrices(net: NetworkClass, k: int, traffic_matrix_list: list, cutoff_path_len=None):
    logger.info("Calculating congestion to all traffic matrices by {} previous average".format(k))

    assert k < len(traffic_matrix_list)
    congestion_ratios = list()
    for index, (current_traffic_matrix, current_opt) in enumerate(traffic_matrix_list[k:]):

        logger.info("Current matrix index is: {}".format(index))
        avg_traffic_matrix = np.mean(list(map(lambda t: t[0], traffic_matrix_list[index:index + k])), axis=0)

        assert avg_traffic_matrix.shape == current_traffic_matrix.shape

        logger.debug("Solving LP problem for previous {} avenge".format(k))
        _, per_edge_flow_fraction_lp = get_optimal_load_balancing(net, avg_traffic_matrix, cutoff_path_len)  # heuristic flows splittings

        logger.debug("Handling the flows that exist in real matrix but not in average one")
        completion_flows_matrix = np.zeros(avg_traffic_matrix.shape)
        flows_to_check = np.dstack(np.where(current_traffic_matrix != 0))[0]
        for src, dst in flows_to_check:
            assert current_traffic_matrix[src][dst] != 0
            completion_flows_matrix[src][dst] = current_traffic_matrix[src][dst] if avg_traffic_matrix[src][dst] == 0 else 0

        # for flows in average is Zero but in real are non zero
        per_edge_flow_fraction_ecmp = get_ecmp_edge_flow_fraction(net, completion_flows_matrix)

        logger.debug("Combining all flows fractions")
        per_edge_flow_fraction = dict()
        for edge, frac_matrix in per_edge_flow_fraction_lp.items():
            if edge in per_edge_flow_fraction_ecmp.keys():
                per_edge_flow_fraction[edge] = frac_matrix + per_edge_flow_fraction_ecmp[edge]
            else:
                per_edge_flow_fraction[edge] = frac_matrix

        for edge, frac_matrix in per_edge_flow_fraction_ecmp.items():
            if edge not in per_edge_flow_fraction.keys():
                per_edge_flow_fraction[edge] = frac_matrix

        logger.debug('Calculating the congestion per edge and finding max edge congestion')

        congestion_per_edge = defaultdict(int)
        max_congestion = 0
        for edge, frac_matrix in per_edge_flow_fraction.items():
            congestion_per_edge[edge] += np.sum(frac_matrix * current_traffic_matrix)
            congestion_per_edge[edge] /= net.get_edge_key(edge=edge, key=EdgeConsts.CAPACITY_STR)
            if congestion_per_edge[edge] > max_congestion:
                max_congestion = congestion_per_edge[edge]

        congestion_ratios.append(max_congestion / current_opt)

    return congestion_ratios


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    c_l = _calculate_congestion_per_matrices(net=net, k=loaded_dict["k"], traffic_matrix_list=loaded_dict["tms"])
    print(np.average(c_l))
