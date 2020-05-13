from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from Learning_to_Route.common.consts import Consts
from flow_routing.find_optimal_load_balancing import *
from ecmp_network import ECMPNetwork, EdgeConsts
from topologies import topologies


def generate_traffic_matrix_baseline(graph: ECMPNetwork,
                                     matrix_sparsity: int, tm_type, elephant_percentage: float,
                                     network_elephant, network_mice, total_matrices: int):
    return [one_sample_tm_base(graph=graph,
                               matrix_sparsity=matrix_sparsity,
                               tm_type=tm_type,
                               elephant_percentage=elephant_percentage, network_elephant=network_elephant,
                               network_mice=network_mice) for _ in range(total_matrices)]


def calculate_congestion_per_matrices(net: ECMPNetwork, k: int, traffic_matrix_list: list):
    assert k < len(traffic_matrix_list)
    congestion_list = list()
    for history_start in range(len(traffic_matrix_list) - k):
        history_end = history_start + k
        current_traffic_matrix = traffic_matrix_list[history_end]
        avg_traffic_matrix = np.mean(traffic_matrix_list[history_start:history_end], axis=0)
        assert avg_traffic_matrix.shape == current_traffic_matrix.shape

        _, per_edge_flow_fraction_lp = get_optimal_load_balancing(net, avg_traffic_matrix)  # heuristic flows splittings

        fixes_matrix = np.zeros(avg_traffic_matrix.shape)
        for i, j in net.get_all_pairs():
            if avg_traffic_matrix[i][j] == 0 and current_traffic_matrix[i][j] != 0:
                fixes_matrix[i][j] = current_traffic_matrix[i][j]

        per_edge_flow_fraction_ecmp = get_ecmp_edge_flow_fraction(net,
                                                                  fixes_matrix)  # for flows in average is Zero but in real are non zero
        per_edge_flow_fraction = dict()
        for edge, frac_matrix in per_edge_flow_fraction_lp.items():
            if edge in per_edge_flow_fraction_ecmp.keys():
                per_edge_flow_fraction[edge] = frac_matrix + per_edge_flow_fraction_ecmp[edge]
            else:
                per_edge_flow_fraction[edge] = frac_matrix

        for edge, frac_matrix in per_edge_flow_fraction_ecmp.items():
            if edge not in per_edge_flow_fraction.keys():
                per_edge_flow_fraction[edge] = frac_matrix

        congestion_per_edge = defaultdict(int)
        max_congestion = 0
        for edge, frac_matrix in per_edge_flow_fraction.items():
            congestion_per_edge[edge] += np.sum(frac_matrix * current_traffic_matrix)
            congestion_per_edge[edge] /= net.get_edge_key(edge=edge, key=EdgeConsts.CAPACITY_STR)
            if congestion_per_edge[edge] > max_congestion:
                max_congestion = congestion_per_edge[edge]

        congestion_list.append(max_congestion)

    return congestion_list


ecmpNetwork = ECMPNetwork(topologies["MESH"])

tms = generate_traffic_matrix_baseline(graph=ecmpNetwork,
                                       matrix_sparsity=0.3, tm_type=Consts.GRAVITY,
                                       elephant_percentage=None, network_elephant=None, network_mice=None,
                                       total_matrices=1000)
c_l = calculate_congestion_per_matrices(net=ecmpNetwork, k=1, traffic_matrix_list=tms)
print(np.average(c_l))
