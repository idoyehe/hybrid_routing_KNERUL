from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass
from consts import EdgeConsts
from generating_tms import load_dump_file
from topologies import topology_zoo_loader
from logger import logger
from argparse import ArgumentParser
from sys import argv
import gurobipy as gb
from gurobipy import GRB


def _oblivious_routing(net: NetworkClass):
    obliv = gb.Model(name="Applegate's and Cohen's Oblivious Routing LP formulations")
    reduced_directed = net.get_graph.to_directed()
    r = obliv.addVar(name="r", lb=0.0, vtype=GRB.CONTINUOUS)
    obliv.setObjective(r, GRB.MINIMIZE)

    pi_edges_dict = defaultdict(dict)
    pe_edges_dict = defaultdict(dict)
    f_arch_dict = defaultdict(dict)

    out_arches_dict = defaultdict(list)
    in_arches_dict = defaultdict(list)
    for _e in net.edges:
        _e_list_sum = 0
        for _h in net.edges:
            pi_edges_dict[_e][_h] = obliv.addVar(name="PI_{}_{}".format(_e, _h), lb=0.0, vtype=GRB.CONTINUOUS)
            cap_h = net.get_edge_key(_h, EdgeConsts.CAPACITY_STR)
            _e_list_sum += cap_h * pi_edges_dict[_e][_h]
        obliv.addConstr(_e_list_sum <= r)

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
                    pe_edges_dict[_e][(i, j)] = obliv.addVar(name="PE_{}_{}".format(_e, (i, j)), lb=0,
                                                             vtype=GRB.CONTINUOUS)

                    f_arch_dict[_arch][(i, j)] = obliv.addVar(name="f_{}_{}".format(_arch, (i, j)), lb=0,
                                                              vtype=GRB.CONTINUOUS)
                    f_arch_dict[_reversed_arch][(i, j)] = obliv.addVar(name="f_{}_{}".format(_reversed_arch, (i, j)),
                                                                       lb=0, vtype=GRB.CONTINUOUS)

                    f_e = f_arch_dict[_arch][(i, j)] + f_arch_dict[_reversed_arch][(i, j)]
                    obliv.addConstr(f_e <= _capacity_e * pe_edges_dict[_e][(i, j)])

        in_arches_dict[_arch[1]].append(_arch)
        out_arches_dict[_arch[0]].append(_arch)

        in_arches_dict[_reversed_arch[1]].append(_reversed_arch)
        out_arches_dict[_reversed_arch[0]].append(_reversed_arch)

    # flow constrains
    for src, dst in net.get_all_pairs():
        assert src != dst
        flow = (src, dst)

        # Flow conservation at the source

        out_flow_from_source = sum([f_arch_dict[out_arch][flow] for out_arch in out_arches_dict[src]])
        in_flow_to_source = sum([f_arch_dict[in_arch][flow] for in_arch in in_arches_dict[src]])
        obliv.addConstr(out_flow_from_source - in_flow_to_source == 1)

        # Flow conservation at the destination

        out_flow_from_dest = sum([f_arch_dict[out_arch][flow] for out_arch in out_arches_dict[dst]])
        in_flow_to_dest = sum([f_arch_dict[in_arch][flow] for in_arch in in_arches_dict[dst]])
        obliv.addConstr(in_flow_to_dest - out_flow_from_dest == 1)

        for u in reduced_directed.nodes:
            if u in flow:
                continue
            # Flow conservation at transit node
            out_flow = sum([f_arch_dict[out_arch][flow] for out_arch in out_arches_dict[u]])
            in_flow = sum([f_arch_dict[in_arch][flow] for in_arch in in_arches_dict[u]])
            obliv.addConstr(out_flow - in_flow == 0)

    for _e in net.edges:
        for i in range(net.get_num_nodes):
            for _arc in reduced_directed.edges:
                _edge_of_arch = _arc
                j = _arc[0]
                k = _arc[1]
                if _edge_of_arch not in list(net.edges):
                    _edge_of_arch = _edge_of_arch[::-1]
                    assert _edge_of_arch in list(net.edges)
                obliv.addConstr(
                    (pi_edges_dict[_e][_edge_of_arch] + pe_edges_dict[_e][(i, j)] - pe_edges_dict[_e][(i, k)]) >= 0)

    logger.info("LP Solving {}".format(obliv.ModelName))
    obliv.optimize()
    if logger.level == logging.DEBUG:
        obliv.printStats()

    per_edge_flow_fraction = dict()
    for _edge in net.edges:
        edge_per_demands = np.zeros((net.get_num_nodes, net.get_num_nodes))
        _arch = _edge
        _reversed_arch = (_edge[1], _edge[0])
        for src, dst in net.get_all_pairs():
            assert src != dst
            edge_per_demands[src, dst] += f_arch_dict[_arch][(src, dst)].x + f_arch_dict[_reversed_arch][(src, dst)].x

        per_edge_flow_fraction[_edge] = edge_per_demands
    return r.x, per_edge_flow_fraction


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
