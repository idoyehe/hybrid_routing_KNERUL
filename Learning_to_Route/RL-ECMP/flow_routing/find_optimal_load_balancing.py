from consts import EdgeConsts
from network_class import NetworkClass, nx
from collections import defaultdict
from logger import *
import numpy as np
import gurobipy as gb
from gurobipy import GRB


def __extract_flows(net: NetworkClass, traffic_demands):
    return list(filter(lambda pair: traffic_demands[pair[0]][pair[1]] > 0, net.get_all_pairs()))


def __validate_solution_on_direct(net: NetworkClass, flows: list, traffic_matrix, out_arches, in_arches,
                                  arch_g_vars_dict):
    for src, dst in flows:
        for v in range(net.get_num_nodes):
            if v == src:
                out_flow_from_source = 0
                for out_arches_from_src in out_arches[v]:
                    out_flow_from_source += arch_g_vars_dict[out_arches_from_src][(src, dst)]
                assert out_flow_from_source == traffic_matrix[src][dst]
                in_flow_to_src = 0
                for in_arches_to_src in in_arches[v]:
                    in_flow_to_src += arch_g_vars_dict[in_arches_to_src][(src, dst)]
                assert in_flow_to_src == 0

            elif v == dst:
                out_flow_from_dst = 0
                for out_arches_from_dst in out_arches[v]:
                    out_flow_from_dst += arch_g_vars_dict[out_arches_from_dst][(src, dst)]
                assert out_flow_from_dst == 0

                in_flow_to_dst = 0
                for in_arches_to_dst in in_arches[v]:
                    in_flow_to_dst += arch_g_vars_dict[in_arches_to_dst][(src, dst)]
                assert in_flow_to_dst == traffic_matrix[src][dst]
            else:
                in_flow_to_v = 0
                for in_arches_to_v in in_arches[v]:
                    in_flow_to_v += arch_g_vars_dict[in_arches_to_v][(src, dst)]
                out_flow_from_v = 0
                for out_arches_from_v in out_arches[v]:
                    out_flow_from_v += arch_g_vars_dict[out_arches_from_v][(src, dst)]
                assert in_flow_to_v == out_flow_from_v


def get_optimal_load_balancing(net: NetworkClass, traffic_demands):
    opt_load = gb.Model(name='Lp for multi commodity flow optimal load balancing')

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = opt_load.addVar(name="r", lb=0.0, vtype=GRB.CONTINUOUS)
    opt_load.setObjective(r, GRB.MINIMIZE)

    arch_g_vars_dict = defaultdict(dict)
    reduced_directed, edge_map_dict, out_arches_dict, in_arches_dict = net.reducing_undirected2directed()

    flows = __extract_flows(net, traffic_demands)
    for u, v, capacity in reduced_directed.edges.data(EdgeConsts.CAPACITY_STR):
        _arch = (u, v)
        all_edge_flow_vars = 0
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)
            g_var = opt_load.addVar(name="{}_g_{}_{}".format(str(_arch), src, dst), lb=0.0, vtype=GRB.CONTINUOUS)
            arch_g_vars_dict[_arch][flow] = g_var
            all_edge_flow_vars += g_var
        if capacity != float("inf"):
            opt_load.addConstr(all_edge_flow_vars <= (capacity * r))

    opt_load.update()

    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        flow = (src, dst)

        # Flow conservation at the source
        out_flow_from_source = sum([arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[src]])
        in_flow_to_source = sum([arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[src]])
        opt_load.addConstr(out_flow_from_source - in_flow_to_source == traffic_demands[src][dst])

        # Flow conservation at the destination
        out_flow_from_dest = sum([arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[dst]])
        in_flow_to_dest = sum([arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[dst]])
        opt_load.addConstr(in_flow_to_dest - out_flow_from_dest == traffic_demands[src][dst])

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = sum([arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[u]])
            in_flow = sum([arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[u]])
            opt_load.addConstr(out_flow - in_flow == 0)

    logger.info("LP Solving {}".format(opt_load.ModelName))
    opt_load.update()

    opt_load.optimize()
    if logger.level == logging.DEBUG:
        opt_load.printStats()
    opt_congestion = r.x

    for _arch, flows_dict in arch_g_vars_dict.items():
        _arch_congestion = reduced_directed.get_edge_key(_arch, EdgeConsts.CAPACITY_STR)
        for (src, dst), var in flows_dict.items():
            flow = (src, dst)
            assert var.x <= traffic_demands[src][dst]
            arch_g_vars_dict[_arch][flow] = var.x

    __validate_solution_on_direct(reduced_directed, flows, traffic_demands, out_arches_dict, in_arches_dict,
                                  arch_g_vars_dict)

    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))

    for edge in reduced_directed.edges:
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)
            per_edge_flow_fraction[edge][src][dst] = float(arch_g_vars_dict[edge][flow] / traffic_demands[src][dst])
    opt_load.close()
    return opt_congestion, per_edge_flow_fraction


def get_ecmp_edge_flow_fraction(net: NetworkClass, traffic_demands):
    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))

    reduced_directed, edge_map_dict, _, _ = net.reducing_undirected2directed()

    logger.info("Handling all flows")
    flows = __extract_flows(net, traffic_demands)
    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        logger.debug("Handle flow form {} to {}".format(src, dst))
        shortest_path_generator = list(reduced_directed.all_shortest_path(source=src, target=dst, weight=None))
        fraction: float = 1.0 / float(len(shortest_path_generator))

        for path in shortest_path_generator:
            for _arch in list(list(map(nx.utils.pairwise, [path]))[0]):
                logger.debug("Handle edge {} in path {}".format(str(_arch), str(path)))
                per_edge_flow_fraction[_arch][src][dst] += fraction

    return per_edge_flow_fraction


if __name__ == "__main__":
    def get_base_graph():
        # init a triangle if we don't get a network graph
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                          (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                          (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 15})])

        return g


    def get_flows_matrix():
        return np.array([[0, 5, 10], [0, 0, 7], [0, 0, 0]])


    ecmpNetwork = NetworkClass(get_base_graph())

    optimal_load_balancing, per_edge_flow_fraction = get_optimal_load_balancing(ecmpNetwork, get_flows_matrix())
    print(optimal_load_balancing)
    print(per_edge_flow_fraction)
