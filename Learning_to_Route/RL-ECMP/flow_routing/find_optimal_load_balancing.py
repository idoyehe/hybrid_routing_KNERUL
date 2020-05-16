from consts import EdgeConsts
from ecmp_network import ECMPNetwork, nx
from collections import defaultdict
from logger import logger
import numpy as np
from docplex.mp.model import Model


def get_optimal_load_balancing(net: ECMPNetwork, traffic_demand, cutoff_path_len=None):
    m = Model(name='Lp for flow load balancing')
    vars_dict = dict()  # dictionary to store all variable problem

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = m.continuous_var(name="r")
    vars_dict["r"] = r

    m.minimize(r)

    vars_per_edge = defaultdict(list)  # dictionary to save all variable related to edge.
    logger.info("LP: Handling all flows")
    for nodes_pair in net.get_all_pairs():
        src, dst = nodes_pair
        src_dest_flow = traffic_demand[src][dst]
        if src_dest_flow > 0:
            logger.debug("Handle flow form {} to {}".format(src, dst))
            vars_per_path = list()
            for path in net.all_simple_paths(source=src, target=dst, cutoff=cutoff_path_len):
                logger.debug("Handle the path {}".format(str(path)))
                var_name = "p_" + '-'.join(str(i) for i in path)
                var = m.continuous_var(lb=0, name=var_name)

                vars_dict[var_name] = var  # adding variable to all vars dict
                vars_per_path.append(var)

                for edge in list(list(map(nx.utils.pairwise, [path]))[0]):
                    logger.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                    if edge[0] > edge[1]:
                        edge = (edge[1], edge[0])
                    vars_per_edge[edge].append((var, nodes_pair))

            m.add_constraint(m.sum(vars_per_path) == src_dest_flow)

    for edge, var_list in vars_per_edge.items():
        m.add_constraint(m.sum([elem[0] for elem in var_list]) <= net.get_edge_key(edge, EdgeConsts.CAPACITY_STR) * r)

    logger.info("LP: Solving")
    m.solve()
    for var_name, var in vars_dict.items():
        logger.debug("Value of variable: {} is: {}".format(var_name, var.solution_value))

    per_edge_flow_fraction = dict()
    for edge, var_list in vars_per_edge.items():
        edge_per_demend = np.zeros((net.get_num_nodes, net.get_num_nodes))
        for var, (src, dst) in var_list:
            edge_per_demend[src][dst] += var.solution_value / traffic_demand[src][dst]
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
# ecmpNetwork = ECMPNetwork(topologies["TRIANGLE"])
# #
# get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())
# get_ecmp_edge_flow_fraction(ecmpNetwork, get_flows_matrix())
