from consts import EdgeConsts
from network_class import NetworkClass, nx
from collections import defaultdict
from logger import *
import numpy as np
from docplex.mp.model import Model


def get_optimal_load_balancing(net: NetworkClass, traffic_demands, cutoff_path_len=None):
    m = Model(name='Lp for flow load balancing')

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = m.continuous_var(name="r")

    m.minimize(r)

    vars_per_edge = defaultdict(list)  # dictionary to save all variable related to edge.
    logger.info("LP: Handling all flows")
    for src, dst in net.get_all_pairs():
        nodes_pair = (src, dst)
        src_dest_flow = traffic_demands[src][dst]
        if src_dest_flow > 0:
            logger.debug("Handle flow form {} to {}".format(src, dst))
            vars_per_flow = list()
            for path in net.all_simple_paths(source=src, target=dst, cutoff=cutoff_path_len):
                logger.debug("Handle the path {}".format(str(path)))
                var = m.continuous_var(lb=0, name="p_" + '>'.join(str(i) for i in path))

                vars_per_flow.append(var)

                for edge in list(list(map(nx.utils.pairwise, [path]))[0]):
                    logger.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                    if edge not in list(net.edges):
                        edge = (edge[1], edge[0])
                    vars_per_edge[edge].append((var, nodes_pair))

            m.add_constraint(m.sum(vars_per_flow) == src_dest_flow)

    for edge, var_list in vars_per_edge.items():
        _capacity_edge = net.get_edge_key(edge, EdgeConsts.CAPACITY_STR)
        m.add_constraint(m.sum([elem[0] for elem in var_list]) <= _capacity_edge * r)

    logger.info("LP Solving {}".format(m.name))
    m.solve()
    if logger.level == logging.DEBUG:
        m.print_solution()

    per_edge_flow_fraction = dict()
    for edge, flow_frac_dict in vars_per_edge.items():
        edge_per_demend = np.zeros((net.get_num_nodes, net.get_num_nodes))
        for var, (src, dst) in flow_frac_dict:
            edge_per_demend[src][dst] += var.solution_value / traffic_demands[src][dst]
        per_edge_flow_fraction[edge] = edge_per_demend

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
                    if edge not in list(net.edges):
                        edge = (edge[1], edge[0])

                    if edge not in per_edge_flow_fraction.keys():
                        per_edge_flow_fraction[edge] = np.zeros((net.get_num_nodes, net.get_num_nodes))
                    per_edge_flow_fraction[edge][src][dst] += fraction

    return per_edge_flow_fraction
