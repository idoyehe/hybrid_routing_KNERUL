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
            g_var = m.continuous_var(name="{}_f_{}_{}".format(str(_arch), src, dst), lb=0, ub=traffic_demands[src][dst])
            arch_g_vars_dict[_arch][flow] = g_var
            all_edge_flow_vars += g_var
        m.add_constraint(all_edge_flow_vars <= capacity * r)

    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        flow = (src, dst)

        # Flow conservation at the source

        out_flow_from_source = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[src]]
        in_flow_to_source = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[src]]
        m.add_constraint(m.sum(out_flow_from_source) == traffic_demands[src][dst])
        m.add_constraint(m.sum(in_flow_to_source) == 0)

        # Flow conservation at the destination

        out_flow_from_dest = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[dst]]
        in_flow_to_dest = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[dst]]
        m.add_constraint((m.sum(in_flow_to_dest) == traffic_demands[src][dst]))
        m.add_constraint(m.sum(out_flow_from_dest) == 0)

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[u]]
            in_flow = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[u]]
            m.add_constraint(m.sum(out_flow) - m.sum(in_flow) == 0)

    logger.info("LP Solving {}".format(m.name))
    m.solve()
    if logger.level == logging.DEBUG:
        m.print_solution()

    for _arch, flows_dict in arch_g_vars_dict.items():
        for (src, dst), var in flows_dict.items():
            flow = (src, dst)
            assert var.solution_value <= traffic_demands[src][dst]
            arch_g_vars_dict[_arch][flow] = var.solution_value / traffic_demands[src][dst]

    __validate_solution_on_direct(reduced_directed, flows, out_arches_dict, in_arches_dict, arch_g_vars_dict)

    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))
    for edge, arch in edge_map_dict.items():
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)

            per_edge_flow_fraction[edge][src][dst] = arch_g_vars_dict[arch][flow]

    return r.solution_value, per_edge_flow_fraction


def get_ecmp_edge_flow_fraction(net: NetworkClass, traffic_demands):
    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))

    logger.info("Handling all flows")
    flows = list(filter(lambda pair: traffic_demands[pair[0]][pair[1]] > 0, net.get_all_pairs()))
    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        flow = (src, dst)
        logger.debug("Handle flow form {} to {}".format(src, dst))
        shortest_path_generator = list(net.all_shortest_path(source=src, target=dst, weight=None))
        fraction: float = 1.0 / float(len(shortest_path_generator))

        for path in shortest_path_generator:
            for edge in list(list(map(nx.utils.pairwise, [path]))[0]):
                logger.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                if edge not in list(net.edges):
                    edge = (edge[1], edge[0])
                per_edge_flow_fraction[edge][src][dst] += fraction

    return per_edge_flow_fraction


def __validate_solution_on_direct(net: NetworkClass, flows: list, out_arches, in_arches, arch_f_vars_dict):
    for src, dst in flows:
        for v in range(net.get_num_nodes):
            if v == src:
                out_flow_from_source = 0
                for out_arches_from_src in out_arches[v]:
                    out_flow_from_source += arch_f_vars_dict[out_arches_from_src][(src, dst)]
                assert out_flow_from_source == 1
                in_flow_to_src = 0
                for in_arches_to_src in in_arches[v]:
                    in_flow_to_src += arch_f_vars_dict[in_arches_to_src][(src, dst)]
                assert in_flow_to_src == 0

            elif v == dst:
                out_flow_from_dst = 0
                for out_arches_from_dst in out_arches[v]:
                    out_flow_from_dst += arch_f_vars_dict[out_arches_from_dst][(src, dst)]
                assert out_flow_from_dst == 0

                in_flow_to_dst = 0
                for in_arches_to_dst in in_arches[v]:
                    in_flow_to_dst += arch_f_vars_dict[in_arches_to_dst][(src, dst)]
                assert in_flow_to_dst == 1
            else:
                in_flow_to_v = 0
                for in_arches_to_v in in_arches[v]:
                    in_flow_to_v += arch_f_vars_dict[in_arches_to_v][(src, dst)]
                out_flow_from_v = 0
                for out_arches_from_v in out_arches[v]:
                    out_flow_from_v += arch_f_vars_dict[out_arches_from_v][(src, dst)]
                assert in_flow_to_v == out_flow_from_v


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
#     return np.array([[0, 5, 10], [0, 0, 7], [0, 0, 0]])
#
#
# ecmpNetwork = NetworkClass(get_base_graph())
#
# solution_value, per_edge_flow_fraction = get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())
# pass
