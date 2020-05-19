from consts import EdgeConsts
from ecmp_network import ECMPNetwork, nx
from collections import defaultdict
from logger import logger
import numpy as np
from docplex.mp.model import Model


def get_optimal_load_balancing(net: ECMPNetwork, traffic_demands):
    m = Model(name='Lp for flow load balancing')
    vars_dict = dict()  # dictionary to store all variable problem

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = m.continuous_var(name="r")
    vars_dict["r"] = r

    m.minimize(r)

    edges_vars_dict = dict()
    reduced_directed, edge_map_dict = net.reducing_undirected2directed()

    out_edge_dict = defaultdict(list)
    in_edge_dict = defaultdict(list)

    for u, v, capacity in reduced_directed.edges.data(EdgeConsts.CAPACITY_STR):
        _edge = (u, v)
        out_edge_dict[u].append(_edge)
        in_edge_dict[v].append(_edge)

        edges_vars_dict[_edge] = []
        _all_edge_vars = []
        for i in range(net.get_num_nodes):
            from_i2j_demends = []
            for j in range(net.get_num_nodes):
                if i == j:
                    from_i2j_demends.append(0)
                    continue
                g_var = m.continuous_var(name="{}_g_{}_{}".format(str(_edge), i, j), lb=0)
                vars_dict[g_var.name] = g_var
                from_i2j_demends.append(g_var)
                _all_edge_vars.append(g_var)
            edges_vars_dict[_edge].append(from_i2j_demends)
        if capacity != float("inf"):
            m.add_constraint(m.sum(_all_edge_vars) <= capacity * r)

    for i in net.get_graph.nodes:  # iterate over original node only
        for j in net.get_graph.nodes:
            if i == j:
                continue
            out_g_i_j = [edges_vars_dict[out_edge][i][j] for out_edge in out_edge_dict[i]]
            in_g_i_j = [edges_vars_dict[in_edge][i][j] for in_edge in in_edge_dict[i]]
            m.add_constraint(m.sum(out_g_i_j) - m.sum(in_g_i_j) == traffic_demands[i][j])

    for k in reduced_directed.nodes:
        for i in net.get_graph.nodes:  # iterate over original node only
            for j in net.get_graph.nodes:
                if i == j or i == k or j == k:
                    continue
                out_g_i_j = [edges_vars_dict[out_edge][i][j] for out_edge in out_edge_dict[k]]
                in_g_i_j = [edges_vars_dict[in_edge][i][j] for in_edge in in_edge_dict[k]]
                m.add_constraint(m.sum(out_g_i_j) - m.sum(in_g_i_j) == 0)

    logger.info("LP: Solving")
    m.solve()
    for var_name, var in vars_dict.items():
        logger.info("Value of variable: {} is: {}".format(var_name, var.solution_value))

    per_edge_flow_fraction = dict()
    for edge, virtual_edge in edge_map_dict.items():
        edge_per_demend = np.zeros((net.get_num_nodes, net.get_num_nodes))
        for src, dst in net.get_all_pairs():
            if traffic_demands[src][dst] != 0:
                edge_per_demend[src][dst] += edges_vars_dict[virtual_edge][src][dst].solution_value / traffic_demands[src][dst]
        per_edge_flow_fraction[edge] = edge_per_demend
    return r.solution_value, per_edge_flow_fraction


def get_ecmp_edge_flow_fraction(net: ECMPNetwork, traffic_demand):
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


# from topologies import topologies
#
#
# def get_flows_matrix():
#     return [[0, 5, 10], [0, 0, 7], [0, 0, 0]]
#
#
# ecmpNetwork = ECMPNetwork(topologies["TRIANGLE"])
# #
# get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())
# get_ecmp_edge_flow_fraction(ecmpNetwork, get_flows_matrix())
