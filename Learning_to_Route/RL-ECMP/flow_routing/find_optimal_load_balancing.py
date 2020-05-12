import pulp as pl
from consts import EdgeConsts
from ecmp_network import ECMPNetwork, nx
from collections import defaultdict
from logger import logger
import numpy as np


def get_optimal_load_balancing(net: ECMPNetwork, traffic_demand):
    lp_problem = pl.LpProblem('Load_Balancing', pl.LpMinimize)
    vars_dict = dict()  # dictionary to store all variable problem

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = pl.LpVariable("r", lowBound=0, upBound=1)
    vars_dict["r"] = r
    lp_problem += r

    vars_per_edge = defaultdict(list)  # dictionary to save all variable related to edge.
    logger.info("Handling all flows")
    for nodes_pair in net.get_all_pairs():
        src, dst = nodes_pair
        src_dest_flow = traffic_demand[src][dst]
        if src_dest_flow > 0:
            logger.debug("Handle flow form {} to {}".format(src, dst))
            vars_per_path = list()
            for path in net.all_simple_paths(source=src, target=dst):
                logger.debug("Handle the path {}".format(str(path)))
                var_name = "x_" + '->'.join(str(i) for i in path)
                var: pl.LpVariable = pl.LpVariable(var_name, lowBound=0)
                vars_dict[var_name] = var  # adding variable to all vars dict
                vars_per_path.append(var)

                for edges_in_path in map(nx.utils.pairwise, [path]):
                    for edge in list(edges_in_path):
                        logger.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                        if edge[0] > edge[1]:
                            edge = (edge[1], edge[0])
                        vars_per_edge[edge].append((var, nodes_pair))

            lp_problem += pl.lpSum(vars_per_path) == src_dest_flow

    for edge, var_list in vars_per_edge.items():
        lp_problem += pl.lpSum([elem[0] for elem in var_list]) <= net.get_edge_key(edge, EdgeConsts.CAPACITY_STR) * r

    status = lp_problem.solve()
    logger.debug(lp_problem)
    logger.info("Status: {}".format(pl.LpStatus[status]))
    for var_name, var in vars_dict.items():
        logger.debug("Value of variable: {} is: {}".format(var_name, pl.value(var)))

    per_edge_flow_fraction = dict()
    for edge, var_list in vars_per_edge.items():
        edge_per_demend = np.zeros((net.get_num_nodes, net.get_num_nodes))
        for var, (src, dst) in var_list:
            edge_per_demend[src][dst] += var.value() / traffic_demand[src][dst]
        per_edge_flow_fraction[edge] = edge_per_demend

    return pl.value(r), per_edge_flow_fraction


def get_base_graph():
    # init a triangle if we don't get a network graph
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                      (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                      (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 15})])

    return g


def get_flows_matrix():
    return [[0, 5, 10], [0, 0, 7], [0, 0, 0]]


ecmpNetwork = ECMPNetwork(get_base_graph())

get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())
