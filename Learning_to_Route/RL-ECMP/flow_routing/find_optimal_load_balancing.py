from consts import EdgeConsts
from network_class import NetworkClass, nx
from collections import defaultdict
from logger import *
import numpy as np
from docplex.mp.model import Model


def __extract_flows(net: NetworkClass, traffic_demands):
    return list(filter(lambda pair: traffic_demands[pair[0]][pair[1]] > 0, net.get_all_pairs()))


def __validate_solution_on_direct(net: NetworkClass, flows: list, traffic_matrix, out_arches, in_arches,
                                  arch_f_vars_dict):
    for src, dst in flows:
        for v in range(net.get_num_nodes):
            if v == src:
                out_flow_from_source = 0
                for out_arches_from_src in out_arches[v]:
                    out_flow_from_source += arch_f_vars_dict[out_arches_from_src][(src, dst)]
                assert out_flow_from_source == traffic_matrix[src][dst]
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
                assert in_flow_to_dst == traffic_matrix[src][dst]
            else:
                in_flow_to_v = 0
                for in_arches_to_v in in_arches[v]:
                    in_flow_to_v += arch_f_vars_dict[in_arches_to_v][(src, dst)]
                out_flow_from_v = 0
                for out_arches_from_v in out_arches[v]:
                    out_flow_from_v += arch_f_vars_dict[out_arches_from_v][(src, dst)]
                assert in_flow_to_v == out_flow_from_v


def __get_optimal_load_balancing(net: NetworkClass, traffic_demands):
    optimal_load = Model(name='Lp for multicommodity flow optimal load balancing')

    # the object variable and function.
    logger.info("Creating linear programing problem")
    r = optimal_load.continuous_var(name="r")
    optimal_load.minimize(r)

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
            g_var = optimal_load.continuous_var(name="{}_f_{}_{}".format(str(_arch), src, dst),
                                                lb=0, ub=traffic_demands[src][dst])
            arch_g_vars_dict[_arch][flow] = g_var
            all_edge_flow_vars += g_var
        if capacity != float("inf"):
            optimal_load.add_constraint(all_edge_flow_vars <= (capacity * r))

    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        flow = (src, dst)

        # Flow conservation at the source
        out_flow_from_source = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[src]]
        in_flow_to_source = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[src]]
        optimal_load.add_constraint(
            optimal_load.sum(out_flow_from_source) - optimal_load.sum(in_flow_to_source) == traffic_demands[src][dst])

        # Flow conservation at the destination
        out_flow_from_dest = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[dst]]
        in_flow_to_dest = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[dst]]
        optimal_load.add_constraint(
            optimal_load.sum(in_flow_to_dest) - optimal_load.sum(out_flow_from_dest) == traffic_demands[src][dst])

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[u]]
            in_flow = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[u]]
            optimal_load.add_constraint(optimal_load.sum(out_flow) - optimal_load.sum(in_flow) == 0)

    logger.info("LP Solving {}".format(optimal_load.name))
    optimal_load.solve()
    if logger.level == logging.DEBUG:
        optimal_load.print_solution()
    for _arch, flows_dict in arch_g_vars_dict.items():
        for (src, dst), var in flows_dict.items():
            flow = (src, dst)
            assert var.solution_value <= traffic_demands[src][dst]
            arch_g_vars_dict[_arch][flow] = var.solution_value
    return r.solution_value


def get_optimal_load_balancing_flows_fractions(net: NetworkClass, traffic_demands):
    opt_congestion = __get_optimal_load_balancing(net, traffic_demands)

    opt_flows = Model(name='Lp for multicommodity flow optimal flow fractions')

    # the object variable and function.
    logger.info("Creating linear programing problem")

    arch_g_vars_dict = defaultdict(dict)
    reduced_directed, edge_map_dict, out_arches_dict, in_arches_dict = net.reducing_undirected2directed()

    flows = __extract_flows(net, traffic_demands)
    all_g_vars_sum = opt_flows.linear_expr()
    for u, v, capacity in reduced_directed.edges.data(EdgeConsts.CAPACITY_STR):
        _arch = (u, v)
        all_edge_flow_vars = 0
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)
            g_var = opt_flows.continuous_var(name="{}_f_{}_{}".format(str(_arch), src, dst), lb=0,
                                             ub=traffic_demands[src][dst])
            arch_g_vars_dict[_arch][flow] = g_var
            all_edge_flow_vars += g_var
            all_g_vars_sum += g_var
        if capacity != float("inf"):
            opt_flows.add_constraint(all_edge_flow_vars <= capacity * opt_congestion)

    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        flow = (src, dst)

        # Flow conservation at the source
        out_flow_from_source = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[src]]
        in_flow_to_source = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[src]]
        opt_flows.add_constraint(
            opt_flows.sum(out_flow_from_source) - opt_flows.sum(in_flow_to_source) == traffic_demands[src][dst])

        # Flow conservation at the destination
        out_flow_from_dest = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[dst]]
        in_flow_to_dest = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[dst]]
        opt_flows.add_constraint(
            opt_flows.sum(in_flow_to_dest) - opt_flows.sum(out_flow_from_dest) == traffic_demands[src][dst])

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = [arch_g_vars_dict[out_arch][flow] for out_arch in out_arches_dict[u]]
            in_flow = [arch_g_vars_dict[in_arch][flow] for in_arch in in_arches_dict[u]]
            opt_flows.add_constraint(opt_flows.sum(out_flow) - opt_flows.sum(in_flow) == 0)

    logger.info("LP Solving {}".format(opt_flows.name))
    opt_flows.minimize(all_g_vars_sum)
    opt_flows.solve()
    if logger.level == logging.DEBUG:
        opt_flows.print_solution()

    max_congestion = -1
    for _arch, flows_dict in arch_g_vars_dict.items():
        _arch_congestion = reduced_directed.get_edge_key(_arch, key=EdgeConsts.CAPACITY_STR)
        _on_arch_congestion = 0
        for (src, dst), var in flows_dict.items():
            flow = (src, dst)
            arch_g_vars_dict[_arch][flow] = var.solution_value
            _on_arch_congestion += var.solution_value
        assert _on_arch_congestion <= _arch_congestion * opt_congestion
        max_congestion = max(max_congestion, _on_arch_congestion / _arch_congestion)

    __validate_solution_on_direct(reduced_directed, flows, traffic_demands,
                                  out_arches_dict, in_arches_dict, arch_g_vars_dict)

    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))

    for edge, arch in edge_map_dict.items():
        for src, dst in flows:
            assert src != dst
            assert traffic_demands[src][dst] > 0
            flow = (src, dst)
            per_edge_flow_fraction[edge][src][dst] = arch_g_vars_dict[arch][flow] / traffic_demands[src][dst]

    return max_congestion, per_edge_flow_fraction


def get_ecmp_edge_flow_fraction(net: NetworkClass, traffic_demands):
    per_edge_flow_fraction = defaultdict(lambda: np.zeros(shape=traffic_demands.shape))

    logger.info("Handling all flows")
    flows = __extract_flows(net, traffic_demands)
    for src, dst in flows:
        assert src != dst
        assert traffic_demands[src][dst] > 0
        logger.debug("Handle flow form {} to {}".format(src, dst))
        shortest_path_generator = list(net.all_shortest_path(source=src, target=dst, weight=None))
        fraction: float = 1.0 / float(len(shortest_path_generator))

        for path in shortest_path_generator:
            for edge in list(list(map(nx.utils.pairwise, [path]))[0]):
                logger.debug("Handle edge {} in path {}".format(str(edge), str(path)))
                if edge not in list(net.edges):
                    edge = edge[::-1]
                per_edge_flow_fraction[edge][src][dst] += fraction

    return per_edge_flow_fraction
