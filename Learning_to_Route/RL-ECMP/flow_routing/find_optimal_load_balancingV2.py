from consts import EdgeConsts
from network_class import NetworkClass, nx
from collections import defaultdict
from logger import *
import numpy as np
from docplex.mp.model import Model


def get_optimal_load_balancing(net: NetworkClass, traffic_demands):
    m = Model(name='Lp for flow load balancing')

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = m.continuous_var(name="r")
    m.minimize(r)

    arch_g_vars_dict = defaultdict(dict)
    reduced_directed, edge_map_dict = net.reducing_undirected2directed()

    out_archs_dict = defaultdict(list)
    in_archs_dict = defaultdict(list)

    for u, v, capacity in reduced_directed.edges.data(EdgeConsts.CAPACITY_STR):
        _arch = (u, v)
        out_archs_dict[u].append(_arch)
        in_archs_dict[v].append(_arch)

        for i in range(net.get_num_nodes):
            for j in range(net.get_num_nodes):
                if i == j:
                    arch_g_vars_dict[_arch][(i, j)] = m.continuous_var(name="{}_g_{}_{}".format(str(_arch), i, j), lb=0, ub=0)
                    continue
                g_var = m.continuous_var(name="{}_g_{}_{}".format(str(_arch), i, j), lb=0)
                arch_g_vars_dict[_arch][(i, j)] = g_var

        if capacity != float("inf"):
            m.add_constraint(m.sum(arch_g_vars_dict[_arch].values()) <= capacity * r)

    for k in reduced_directed.nodes:
        for i in net.get_graph.nodes:  # iterate over original node only
            for j in net.get_graph.nodes:
                if i == j or j == k:
                    continue

                out_g_k_i_j = [arch_g_vars_dict[out_arch][(i, j)] for out_arch in out_archs_dict[k]]
                in_g_k_i_j = [arch_g_vars_dict[in_arch][(i, j)] for in_arch in in_archs_dict[k]]
                if i == k:
                    m.add_constraint(m.sum(out_g_k_i_j) - m.sum(in_g_k_i_j) == traffic_demands[i][j])
                else:
                    m.add_constraint(m.sum(out_g_k_i_j) - m.sum(in_g_k_i_j) == 0)

    logger.info("LP: Solving")
    m.solve()
    if logger.level == logging.DEBUG:
        m.print_solution()

    per_edge_flow_fraction = dict()
    for edge, virtual_edge in edge_map_dict.items():
        edge_per_demands = np.zeros((net.get_num_nodes, net.get_num_nodes))
        for (src, dst), var in arch_g_vars_dict[virtual_edge].items():
            if traffic_demands[src][dst] > 0:
                edge_per_demands[src][dst] += var.solution_value / traffic_demands[src][dst]

        per_edge_flow_fraction[edge] = edge_per_demands
    return r.solution_value, per_edge_flow_fraction


def get_ecmp_edge_flow_fraction(net: NetworkClass, traffic_demand):
    per_edge_flow_fraction = dict()

    logger.info("Handling all flows")
    for nodes_pair in net.get_all_pairs():
        src, dst = nodes_pair
        src_dest_flow = traffic_demand[src][dst]
        if src_dest_flow > 0:
            logger.debug("Handle flow form {} to {}".format(src, dst))
            shortest_path_generator = list(net.all_shortest_path(source=src, target=dst, weight=None))
            fraction = 1 / len(shortest_path_generator)

            for path in shortest_path_generator:
                for edge in list(list(map(nx.utils.pairwise, [path]))[0]):
                    logger.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                    if edge[0] > edge[1]:
                        edge = (edge[1], edge[0])

                    if edge not in per_edge_flow_fraction.keys():
                        per_edge_flow_fraction[edge] = np.zeros((net.get_num_nodes, net.get_num_nodes))
                    per_edge_flow_fraction[edge][src][dst] += fraction

    return per_edge_flow_fraction


from topologies import topologies


def get_flows_matrix():
    return [[0, 5, 10], [0, 0, 7], [0, 0, 0]]


ecmpNetwork = NetworkClass(topologies["TRIANGLE"])
#
get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())
get_ecmp_edge_flow_fraction(ecmpNetwork, get_flows_matrix())
