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

    arch_f_vars_dict = defaultdict(dict)
    reduced_directed, edge_map_dict = net.reducing_undirected2directed()

    out_arches_dict = defaultdict(list)
    in_arches_dict = defaultdict(list)

    flows = list(filter(lambda pair: traffic_demands[pair[0]][pair[1]] > 0, net.get_all_pairs()))

    for u, v, capacity in reduced_directed.edges.data(EdgeConsts.CAPACITY_STR):
        _arch = (u, v)
        out_arches_dict[u].append(_arch)
        in_arches_dict[v].append(_arch)
        all_edge_flow_vars = 0
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)
            f_var = m.continuous_var(name="{}_f_{}_{}".format(str(_arch), src, dst), lb=0, ub=1)
            arch_f_vars_dict[_arch][flow] = f_var
            all_edge_flow_vars += traffic_demands[src][dst] * f_var
        m.add_constraint(all_edge_flow_vars <= capacity * r)

    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        flow = (src, dst)

        # Flow conservation at the source

        out_flow_origin_source = [arch_f_vars_dict[out_arch][flow] for out_arch in out_arches_dict[dst]]
        in_flow_origin_source = [arch_f_vars_dict[in_arch][flow] for in_arch in in_arches_dict[dst]]
        m.add_constraint(m.sum(out_flow_origin_source) - m.sum(in_flow_origin_source) == 1)

        # Flow conservation at the destination

        out_flow_to_dest = [arch_f_vars_dict[out_arch][flow] for out_arch in out_arches_dict[src]]
        in_flow_to_dest = [arch_f_vars_dict[in_arch][flow] for in_arch in in_arches_dict[src]]
        m.add_constraint(m.sum(in_flow_to_dest) - m.sum(out_flow_to_dest) == 1)

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = [arch_f_vars_dict[out_arch][flow] for out_arch in out_arches_dict[u]]
            in_flow = [arch_f_vars_dict[in_arch][flow] for in_arch in in_arches_dict[u]]
            m.add_constraint(m.sum(out_flow) - m.sum(in_flow) == 0)

    logger.info("LP Solving {}".format(m.name))
    m.solve()
    if logger.level == logging.DEBUG:
        m.print_solution()

    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))
    for edge, arch in edge_map_dict.items():
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)

            per_edge_flow_fraction[edge][src][dst] = arch_f_vars_dict[arch][flow]

    return r.solution_value, per_edge_flow_fraction


def get_ecmp_edge_flow_fraction(net: NetworkClass, traffic_demand):
    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demand.shape))

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
                    per_edge_flow_fraction[edge][src][dst] += fraction

    return per_edge_flow_fraction
