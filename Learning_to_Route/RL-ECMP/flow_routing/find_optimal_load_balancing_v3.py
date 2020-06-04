from consts import EdgeConsts
from network_class import NetworkClass, nx
from collections import defaultdict
from logger import *
import numpy as np
import gurobipy as gb
from gurobipy import GRB


def __extract_flows(net: NetworkClass, traffic_demands):
    return list(filter(lambda pair: traffic_demands[pair[0]][pair[1]] > 0, net.get_all_pairs()))


def __validate_solution_on_direct(net_directed: NetworkClass, flows: list, traffic_matrix, out_arches, in_arches,
                                  arch_g_vars_dict):
    for flow in flows:
        for v in range(net_directed.get_num_nodes):
            if v == flow[0]:
                from_its_source = 0
                for out_arches_from_src in out_arches[v]:
                    from_its_source += arch_g_vars_dict[out_arches_from_src][flow]
                assert from_its_source == traffic_matrix[flow]
                to_its_src = 0
                for in_arches_to_src in in_arches[v]:
                    to_its_src += arch_g_vars_dict[in_arches_to_src][flow]
                assert to_its_src == 0

            elif v == flow[1]:
                from_its_dst = 0
                for out_arches_from_dst in out_arches[v]:
                    from_its_dst += arch_g_vars_dict[out_arches_from_dst][flow]
                assert from_its_dst == 0

                to_its_dst = 0
                for in_arches_to_dst in in_arches[v]:
                    to_its_dst += arch_g_vars_dict[in_arches_to_dst][flow]
                assert to_its_dst == traffic_matrix[flow]
            else:
                assert v not in flow
                to_some_v = 0
                for in_arches_to_v in in_arches[v]:
                    to_some_v += arch_g_vars_dict[in_arches_to_v][flow]
                from_some_v = 0
                for out_arches_from_v in out_arches[v]:
                    from_some_v += arch_g_vars_dict[out_arches_from_v][flow]
                assert to_some_v == from_some_v


def optimal_load_balancing_finder(net: NetworkClass, traffic_matrix, opt_ratio_value=None):
    opt_lp_problem = gb.Model(name="LP problem for optimal load balancing, given network and TM")

    flows = __extract_flows(net, traffic_matrix)

    arch_vars_per_flow = defaultdict(dict)
    arch_all_vars = defaultdict(list)

    net_direct, edge2v_edge_dict, out_arches_dict, in_arches_dict = net.reducing_undirected2directed()
    for _arch in net_direct.edges:
        for flow in flows:
            src, dst = flow
            assert src != dst
            assert traffic_matrix[flow] > 0

            g_var = opt_lp_problem.addVar(lb=0.0, name="g_{}_{}->{}".format(str(_arch), src, dst), vtype=GRB.CONTINUOUS)
            arch_vars_per_flow[_arch][flow] = g_var
            arch_all_vars[_arch].append(g_var)

    if opt_ratio_value is None:
        opt_ratio = opt_lp_problem.addVar(lb=0.0, name="opt_ratio", vtype=GRB.CONTINUOUS)
        opt_lp_problem.setObjective(opt_ratio,GRB.MINIMIZE)
    else:
        opt_ratio = opt_ratio_value

    for _arch in net_direct.edges:
        _arch_capacity = net_direct.get_edge_key(_arch, key=EdgeConsts.CAPACITY_STR)
        opt_lp_problem.addConstr(sum(arch_all_vars[_arch]) <= _arch_capacity * opt_ratio)

    for flow in flows:
        src, dst = flow
        assert src != dst
        assert traffic_matrix[flow] > 0

        # Flow conservation at the source
        from_its_src = sum([arch_vars_per_flow[out_arch][flow] for out_arch in out_arches_dict[src]])
        to_its_src = sum([arch_vars_per_flow[in_arch][flow] for in_arch in in_arches_dict[src]])
        opt_lp_problem.addConstr((from_its_src - to_its_src) == traffic_matrix[src][dst])
        # opt_lp_problem.add_constraint(to_its_src == 0)

        # Flow conservation at the destination
        from_its_dst = sum([arch_vars_per_flow[out_arch][flow] for out_arch in out_arches_dict[dst]])
        to_its_dst = sum([arch_vars_per_flow[in_arch][flow] for in_arch in in_arches_dict[dst]])
        opt_lp_problem.addConstr((to_its_dst - from_its_dst) == traffic_matrix[src][dst])
        # opt_lp_problem.add_constraint(from_its_dst == 0)

        for u in net_direct.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            from_some_u = sum([arch_vars_per_flow[out_arch][flow] for out_arch in out_arches_dict[u]])
            to_some_u = sum([arch_vars_per_flow[in_arch][flow] for in_arch in in_arches_dict[u]])
            opt_lp_problem.addConstr(from_some_u == to_some_u)

    logger.info("LP Solving {}".format(opt_lp_problem.ModelName))
    opt_lp_problem.optimize()

    if opt_ratio_value is None:
        opt_ratio = opt_ratio.x

    if logger.level == logging.DEBUG:
        opt_lp_problem.printStats()

    for _arch in net_direct.edges:
        for flow in flows:
            src, dst = flow
            assert src != dst
            assert traffic_matrix[flow] > 0
            arch_vars_per_flow[_arch][flow] = arch_vars_per_flow[_arch][flow].x

    opt_lp_problem.close()
    __validate_solution_on_direct(net_direct, flows, traffic_matrix, out_arches_dict, in_arches_dict,
                                  arch_vars_per_flow)

    link_carries_per_flow = defaultdict(lambda: np.zeros(shape=traffic_matrix.shape, dtype=np.float64))
    for real_link, virtual_link in edge2v_edge_dict.items():
        for flow in flows:
            src, dst = flow
            flow_demand = traffic_matrix[flow]
            assert src != dst
            assert flow_demand > 0
            link_carries_per_flow[real_link][flow] += float(arch_vars_per_flow[virtual_link][flow]) / float(flow_demand)

    max_congested_link = 0

    for u, v, link_capacity in net.edges.data(EdgeConsts.CAPACITY_STR):
        link = (u, v)
        fractions_from_lp = link_carries_per_flow[link]
        total_link_load = np.sum(np.multiply(fractions_from_lp, traffic_matrix))
        max_congested_link = max(max_congested_link, (float(total_link_load) / float(link_capacity)))

    assert np.abs(max_congested_link - opt_ratio) <= float(1e-8)
    return max_congested_link, link_carries_per_flow


def get_ecmp_edge_flow_fraction(net: NetworkClass, traffic_demands, weight=None):
    per_edge_flow_fraction = defaultdict(int)

    logger.info("Handling all flows")
    flows = __extract_flows(net, traffic_demands)
    for flow in flows:
        src, dst = flow
        assert traffic_demands[flow] > 0
        logger.debug("Handle flow form {} to {}".format(src, dst))
        shortest_path_generator = list(net.all_shortest_path(source=src, target=dst, weight=weight))
        total_paths = float(len(shortest_path_generator))
        flow_fraction: float = traffic_demands[flow] / total_paths

        for _path in shortest_path_generator:
            for _arch in list(list(map(nx.utils.pairwise, [_path]))[0]):
                logger.debug("Handle edge {} in path {}".format(str(_arch), str(_path)))
                if _arch not in list(net.edges):
                    _arch = _arch[::-1]
                per_edge_flow_fraction[_arch] += flow_fraction

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


    net = NetworkClass(get_base_graph())

    optimal_load_balancing, per_edge_flow_fraction = optimal_load_balancing_finder(net, get_flows_matrix())
    per_edge_ecmp_flow_fraction = get_ecmp_edge_flow_fraction(net, get_flows_matrix())
    print(optimal_load_balancing)
    print(per_edge_flow_fraction)
    print(per_edge_ecmp_flow_fraction)
