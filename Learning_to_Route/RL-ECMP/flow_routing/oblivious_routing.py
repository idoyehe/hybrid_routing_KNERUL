from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from Learning_to_Route.common.consts import Consts
from flow_routing.find_optimal_load_balancing import *
from ecmp_network import ECMPNetwork, EdgeConsts
from topologies import topologies, topology_zoo_loader
from docplex.mp.model import Model


def _set_polyhedron_h_1(net: ECMPNetwork, model: Model):
    traffic_demands = [
        [model.continuous_var(name="D_{}_{}".format(i, j), lb=0) if i != j else 0 for j in range(net.get_num_nodes)]
        for i in range(net.get_num_nodes)]

    edges_vars_dict = dict()
    for edge in net.get_graph.edges:
        edges_vars_dict[edge] = [
            [model.continuous_var(name="{}_g_{}_{}".format(str(edge), i, j), lb=0) if i != j else 0 for j in range(net.get_num_nodes)]
            for i in range(net.get_num_nodes)]
        flat_list = [item for sublist in edges_vars_dict[edge] for item in sublist]

        m.add_constraint(m.sum(flat_list) <= net.get_edge_key(edge,EdgeConsts.CAPACITY_STR))

    for node in net.get_graph.nodes:
        node_out_edges = list()
        node_in_edges = list()
        for src, dst in net.get_graph.out_edges:
            if src == node:
                node_out_edges.append((src, dst))
            else:
                continue
        for src, dst in net.get_graph.in_edges:
            if dst == node:
                node_in_edges.append((src, dst))
            else:
                continue
        for dst in net.get_graph.nodes:
            if dst == node: continue
            out_vars = list()
            in_vars = list()
            for out_edge in node_out_edges:
                out_vars.append(edges_vars_dict[out_edge][node][dst])
            for in_edge in node_in_edges:
                if dst == node: continue
                in_vars.append(edges_vars_dict[in_edge][node][dst])
            m.add_constraint(m.sum(out_vars) - m.sum(in_vars) == traffic_demands[node][dst])

    for k in net.get_graph.nodes:
        node_out_edges = list()
        node_in_edges = list()
        for src, dst in net.get_graph.out_edges:
            if src == k:
                node_out_edges.append((src, dst))
            else:
                continue
        for src, dst in net.get_graph.in_edges:
            if dst == k:
                node_in_edges.append((src, dst))
            else:
                continue

        for src in net.get_graph.nodes:
            for dst in net.get_graph.nodes:
                if src == dst or src == k or dst == k:
                    continue
                out_vars = list()
                in_vars = list()

                for out_edge in node_out_edges:
                    out_vars.append(edges_vars_dict[out_edge][src][dst])
                for in_edge in node_in_edges:
                    in_vars.append(edges_vars_dict[in_edge][src][dst])
                m.add_constraint(m.sum(out_vars) - m.sum(in_vars) == 0)


    r = m.continuous_var(name="r")
    m.maximize(r)
    for edge, var_values in edges_vars_dict.items():
        flat_list = [item for sublist in var_values for item in sublist]
        m.add_constraint(m.sum(flat_list) <= r*net.get_edge_key(edge=edge,key=EdgeConsts.CAPACITY_STR))

    m.solve()
    m.print_information()
    m.print_solution()





    return traffic_demands


# def calculate_congestion_per_matrices(net: ECMPNetwork, k: int, traffic_matrix_list: list, cutoff_path_len=None):
#     logger.info("Calculating congestion to all traffic matrices by {} previous average".format(k))
#
#     assert k < len(traffic_matrix_list)
#     congestion_list = list()
#     for index, current_traffic_matrix in enumerate(traffic_matrix_list[k:]):
#
#         logger.info("Current matrix index is: {}".format(index))
#         avg_traffic_matrix = np.mean(traffic_matrix_list[index:index + k], axis=0)
#
#         assert avg_traffic_matrix.shape == current_traffic_matrix.shape
#
#         logger.debug("Solving LP problem for previous {} avenge".format(k))
#         _, per_edge_flow_fraction_lp = get_optimal_load_balancing(net, avg_traffic_matrix, cutoff_path_len)  # heuristic flows splittings
#
#         logger.debug("Handling the flows that exist in real matrix but not in average one")
#         completion_flows_matrix = np.zeros(avg_traffic_matrix.shape)
#         flows_to_check = np.dstack(np.where(current_traffic_matrix != 0))[0]
#         for src, dst in flows_to_check:
#             assert current_traffic_matrix[src][dst] != 0
#             completion_flows_matrix[src][dst] = current_traffic_matrix[src][dst] if avg_traffic_matrix[src][dst] == 0 else 0
#
#         # for flows in average is Zero but in real are non zero
#         per_edge_flow_fraction_ecmp = get_ecmp_edge_flow_fraction(net, completion_flows_matrix)
#
#         logger.debug("Combining all flows fractions")
#         per_edge_flow_fraction = dict()
#         for edge, frac_matrix in per_edge_flow_fraction_lp.items():
#             if edge in per_edge_flow_fraction_ecmp.keys():
#                 per_edge_flow_fraction[edge] = frac_matrix + per_edge_flow_fraction_ecmp[edge]
#             else:
#                 per_edge_flow_fraction[edge] = frac_matrix
#
#         for edge, frac_matrix in per_edge_flow_fraction_ecmp.items():
#             if edge not in per_edge_flow_fraction.keys():
#                 per_edge_flow_fraction[edge] = frac_matrix
#
#         logger.debug('Calculating the congestion per edge and finding max edge congestion')
#
#         congestion_per_edge = defaultdict(int)
#         max_congestion = 0
#         opt, _ = get_optimal_load_balancing(net, current_traffic_matrix, cutoff_path_len)  # heuristic flows splittings
#         for edge, frac_matrix in per_edge_flow_fraction.items():
#             congestion_per_edge[edge] += np.sum(frac_matrix * current_traffic_matrix)
#             congestion_per_edge[edge] /= net.get_edge_key(edge=edge, key=EdgeConsts.CAPACITY_STR)
#             if congestion_per_edge[edge] > max_congestion:
#                 max_congestion = congestion_per_edge[edge]
#
#         congestion_list.append(max_congestion/opt)
#
#     return congestion_list


# K = 3
# ecmpNetwork = ECMPNetwork(topology_zoo_loader("http://www.topology-zoo.org/files/Ibm.gml", default_capacity=45))
# average_capacity = np.mean(list(ecmpNetwork.get_edges_capacities().values()))
# tms = generate_traffic_matrix_baseline(graph=ecmpNetwork,
#                                        k=K,
#                                        matrix_sparsity=0.3,
#                                        tm_type=Consts.GRAVITY,
#                                        elephant_percentage=0.2, network_elephant=average_capacity, network_mice=average_capacity * 0.1,
#                                        total_matrices=100)
# c_l = calculate_congestion_per_matrices(net=ecmpNetwork, k=K, traffic_matrix_list=tms)
# print(np.average(c_l))

def _triangle():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                      (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                      (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 15})])

    return g.to_directed()


ecmpNetwork = ECMPNetwork(_triangle())
m = Model(name='Lp for oblivious routing')

_set_polyhedron_h_1(ecmpNetwork, m)
