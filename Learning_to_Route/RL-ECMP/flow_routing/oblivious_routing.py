from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass
from consts import EdgeConsts
from generating_tms import load_dump_file
from topologies import topology_zoo_loader
from logger import logger
from argparse import ArgumentParser
from sys import argv


def _oblivious_routing(net: NetworkClass):
    m = Model(name="Applegate's and Cohen's Oblivious Routing LP formulations")
    reduced_directed = net.get_graph.to_directed()
    r = m.continuous_var(name="r")
    m.minimize(r)

    pi_edges_dict = defaultdict(dict)
    pe_edges_dict = defaultdict(dict)
    f_arch_dict = defaultdict(dict)

    out_arches = defaultdict(list)
    in_arches = defaultdict(list)
    for _e in net.edges:
        _e_list_sum = []
        for _h in net.edges:
            pi_edges_dict[_e][_h] = m.continuous_var(name="PI_{}_{}".format(_e, _h), lb=0)
            cap_h = net.get_edge_key(_h, EdgeConsts.CAPACITY_STR)
            _e_list_sum.append(cap_h * pi_edges_dict[_e][_h])
        m.add_constraint(m.sum(_e_list_sum) <= r)

    for _e in net.edges:
        _capacity_e = net.get_edge_key(_e, EdgeConsts.CAPACITY_STR)
        _arch = _e
        _reversed_arch = (_e[1], _e[0])
        for i in range(net.get_num_nodes):
            for j in range(net.get_num_nodes):
                if i == j:
                    pe_edges_dict[_e][(i, j)] = 0
                    f_arch_dict[_arch][(i, j)] = 0
                    f_arch_dict[_reversed_arch][(i, j)] = 0
                else:
                    pe_edges_dict[_e][(i, j)] = m.continuous_var(name="PE_{}_{}".format(_e, (i, j)), lb=0)
                    f_arch_dict[_arch][(i, j)] = m.continuous_var(name="f_{}_{}".format(_arch, (i, j)), lb=0)
                    f_arch_dict[_reversed_arch][(i, j)] = m.continuous_var(
                        name="f_{}_{}".format(_reversed_arch, (i, j)), lb=0)

                    f_e = f_arch_dict[_arch][(i, j)] + f_arch_dict[_reversed_arch][(i, j)]
                    m.add_constraint(f_e / _capacity_e <= pe_edges_dict[_e][(i, j)])

        in_arches[_arch[1]].append(_arch)
        out_arches[_arch[0]].append(_arch)

        in_arches[_reversed_arch[1]].append(_reversed_arch)
        out_arches[_reversed_arch[0]].append(_reversed_arch)

    # flow constrains
    for src, dst in net.get_all_pairs():
        assert src != dst
        flow = (src, dst)

        # Flow conservation at the source
        out_flow_origin_source = [f_arch_dict[out_arch][flow] for out_arch in out_arches[src]]
        in_flow_origin_source = [f_arch_dict[in_arch][flow] for in_arch in in_arches[src]]
        m.add_constraint(m.sum(out_flow_origin_source) == 1)
        m.add_constraint(m.sum(in_flow_origin_source) == 0)

        # Flow conservation at the destination
        out_flow_to_dest = [f_arch_dict[out_arch][flow] for out_arch in out_arches[dst]]
        in_flow_to_dest = [f_arch_dict[in_arch][flow] for in_arch in in_arches[dst]]
        m.add_constraint((m.sum(in_flow_to_dest) == 1))
        m.add_constraint(m.sum(out_flow_to_dest) == 0)

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = [f_arch_dict[out_arch][flow] for out_arch in out_arches[u]]
            in_flow = [f_arch_dict[in_arch][flow] for in_arch in in_arches[u]]
            m.add_constraint(m.sum(out_flow) - m.sum(in_flow) == 0)

    for _e in net.edges:
        for i in range(net.get_num_nodes):
            for _arc in reduced_directed.edges:
                _edge_of_arch = _arc
                j = _arc[0]
                k = _arc[1]
                if _edge_of_arch not in list(net.edges):
                    _edge_of_arch = (_arc[1], _arc[0])
                    assert _edge_of_arch in net.edges

                m.add_constraint(
                    (pi_edges_dict[_e][_edge_of_arch] + pe_edges_dict[_e][(i, j)] - pe_edges_dict[_e][(i, k)]) >= 0)

    logger.info("LP Solving {}".format(m.name))
    m.solve()
    if logger.level == logging.DEBUG:
        m.print_solution()

    per_edge_flow_fraction = dict()
    for _edge in net.edges:
        edge_per_demands = np.zeros((net.get_num_nodes, net.get_num_nodes))
        _arch = _edge
        _reversed_arch = (_edge[1], _edge[0])
        for src, dst in net.get_all_pairs():
            assert src != dst
            edge_per_demands[src, dst] += f_arch_dict[_arch][(src, dst)].solution_value + f_arch_dict[_reversed_arch][
                (src, dst)].solution_value

        per_edge_flow_fraction[_edge] = edge_per_demands
    return r.solution_value, per_edge_flow_fraction


def _calculate_congestion_per_matrices(net: NetworkClass, traffic_matrix_list: list, oblivious_routing_per_edge: dict):
    logger.info("Calculating congestion to all traffic matrices by {} oblivious routing")

    congestion_ratios = list()
    for index, (current_traffic_matrix, current_opt) in enumerate(traffic_matrix_list):
        logger.info("Current matrix index is: {}".format(index))

        assert current_traffic_matrix.shape == (net.get_num_nodes, net.get_num_nodes)

        logger.debug('Calculating the congestion per edge and finding max edge congestion')

        congestion_per_edge = defaultdict(int)
        max_congestion = 0
        for edge, frac_matrix in oblivious_routing_per_edge.items():
            congestion_per_edge[edge] += np.sum(frac_matrix * current_traffic_matrix)
            congestion_per_edge[edge] /= net.get_edge_key(edge=edge, key=EdgeConsts.CAPACITY_STR)
            if congestion_per_edge[edge] > max_congestion:
                max_congestion = congestion_per_edge[edge]

        assert max_congestion >= current_opt
        congestion_ratios.append(max_congestion / current_opt)

    return congestion_ratios


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    dump_path = _getOptions().dumped_path
    loaded_dict = load_dump_file(dump_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    oblivious_ratio, oblivious_routing_per_edge = _oblivious_routing(net)
    print("The oblivious ratio for {} is {}".format(net.get_name, oblivious_ratio))
    c_l = _calculate_congestion_per_matrices(net=net, traffic_matrix_list=loaded_dict["tms"],
                                             oblivious_routing_per_edge=oblivious_routing_per_edge)
    print(np.average(c_l))

