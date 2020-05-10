import pulp as pl
from consts import EdgeConsts
from ecmp_network import ECMPNetwork, nx
from collections import defaultdict
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def get_optimal_load_balancing(net: ECMPNetwork, traffic_demand):
    lp_problem = pl.LpProblem('Load_Balancing', pl.LpMinimize)
    vars_dict = dict()  # dictionary to store all variable problem

    # the object variable and function.
    logging.info("Creating linear programing problem")
    r = pl.LpVariable("r", lowBound=0, upBound=1)
    vars_dict["r"] = r
    lp_problem += r

    vars_per_edge = defaultdict(list)  # dictionary to save all variable related to edge.
    logging.info("Handling all flows")
    for nodes_pair in net.get_all_pairs():
        src, dst = nodes_pair
        src_dest_flow = traffic_demand[src][dst]
        if src_dest_flow > 0:
            logging.debug("Handle flow form {} to {}".format(src, dst))
            vars_per_path = list()
            for path in net.all_simple_paths(source=src, target=dst):
                logging.debug("Handle the path {}".format(str(path)))
                var_name = "x_" + '->'.join(str(i) for i in path)
                var: pl.LpVariable = pl.LpVariable(var_name, lowBound=0)
                vars_dict[var_name] = var  # adding variable to all vars dict
                vars_per_path.append(var)

                for edges_in_path in map(nx.utils.pairwise, [path]):
                    for edge in list(edges_in_path):
                        logging.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                        if edge[0] > edge[1]:
                            edge = (edge[1], edge[0])
                        vars_per_edge[edge].append(var)

            lp_problem += pl.lpSum(vars_per_path) == src_dest_flow

    for edge, var_list in vars_per_edge.items():
        lp_problem += pl.lpSum(var_list) <= net.get_edge_key(edge, EdgeConsts.CAPACITY_STR) * r

    status = lp_problem.solve()
    logging.debug(lp_problem)
    logging.info("Status: {}".format(pl.LpStatus[status]))
    for var_name, var in vars_dict.items():
        logging.debug("Value of variable: {} is: {}".format(var_name, pl.value(var)))

    return pl.value(r)



# def get_base_graph():
#     # init a triangle if we don't get a network graph
#     g = nx.Graph()
#     g.add_nodes_from([0, 1, 2])
#     g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
#                       (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
#                       (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 15})])
#
#     return g
#
#
# def get_flows_matrix():
#     return [[0, 5, 10], [0, 0, 7], [0, 0, 0]]
#
#
# ecmpNetwork = ECMPNetwork(get_base_graph())
#
# get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())

