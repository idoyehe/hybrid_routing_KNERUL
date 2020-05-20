from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from Learning_to_Route.common.consts import Consts
from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass, EdgeConsts
from topologies import topology_zoo_loader
import pickle


def generate_traffic_matrix_baseline(net: NetworkClass, k: int,
                                     matrix_sparsity: float, tm_type, elephant_percentage: float,
                                     network_elephant, network_mice, total_matrices: int):
    logger.info("Generating baseline of traffic matrices to evaluate of length {}".format(total_matrices + k))
    tm_list = list()
    for _ in range(total_matrices + k):
        tm = one_sample_tm_base(graph=net,
                                matrix_sparsity=matrix_sparsity,
                                tm_type=tm_type,
                                elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                                network_mice=network_mice)
        opt, _ = get_optimal_load_balancing(net, tm, None)  # heuristic flows splittings
        tm_list.append((tm, opt))

    return tm_list


def calculate_congestion_per_matrices(net: NetworkClass, k: int, traffic_matrix_list: list, from_dumped=False, cutoff_path_len=None):
    logger.info("Calculating congestion to all traffic matrices by {} previous average".format(k))

    assert k < len(traffic_matrix_list)
    congestion_list = list()
    for index, (current_traffic_matrix, current_opt) in enumerate(traffic_matrix_list[k:]):

        logger.info("Current matrix index is: {}".format(index))
        avg_traffic_matrix = np.mean(traffic_matrix_list[index:index + k], axis=0)

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

        congestion_list.append(max_congestion / current_opt)

    return congestion_list


def dump_tms_and_opt(net: NetworkClass, k: int, matrix_sparsity: float, tm_type, elephant_percentage: float,
                     network_elephant, network_mice, total_matrices: int):
    tms = generate_traffic_matrix_baseline(net=net,
                                           k=k, matrix_sparsity=matrix_sparsity, tm_type=tm_type,
                                           elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                                           network_mice=network_mice,
                                           total_matrices=total_matrices)
    file_name: str = "C:\\Users\\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\RL-ECMP\\tm_dumps\\{}_tms_{}X{}_length_{}_{}_sparsity_{}".format(
        net.get_name, net.get_num_nodes, net.get_num_nodes, k, tm_type, matrix_sparsity)
    dump_file = open(file_name, 'wb')
    pickle.dump(tms, dump_file)
    dump_file.close()
    return file_name


def load_tms_and_opt(file_name: str):
    dumped_file = open(file_name, 'rb')
    tms = pickle.load(dumped_file)
    dumped_file.close()
    assert isinstance(tms, list)
    return tms


if __name__ == "__main__":
    net = NetworkClass(topology_zoo_loader("http://www.topology-zoo.org/files/Ibm.gml", default_capacity=45))
    average_capacity = np.mean(list(net.get_edges_capacities().values()))
    K = 3
    filename: str = dump_tms_and_opt(net=net, k=K,
                                     matrix_sparsity=0.3,
                                     tm_type=Consts.GRAVITY,
                                     elephant_percentage=0.2,
                                     network_elephant=average_capacity,
                                     network_mice=average_capacity * 0.1,
                                     total_matrices=1)

    _tms = load_tms_and_opt(filename)
    c_l = calculate_congestion_per_matrices(net=net, k=K, traffic_matrix_list=_tms)
    print(np.average(c_l))
