import numpy as np
from argparse import ArgumentParser
from sys import argv
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.consts import EdgeConsts


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for console output file")
    parser.add_argument("-o_p", "--oblivious_path", type=str, help="The path for oblivious routing schemes")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from")
    parser.add_argument("-rl_p", "--learning_path", type=str, help="The path for RL routing schemes")
    options = parser.parse_args(args)
    return options


def oblivious_parsing(oblivious_path):
    oblivious_matrices_file = open(oblivious_path, "rb")
    oblivious_matrices = np.load(oblivious_matrices_file)
    gravity_traffic = net.gravity_traffic_map()
    gravity_tm = np.zeros((net.get_num_nodes, net.get_num_nodes), dtype=np.float32)
    for src, dst, demand in gravity_traffic:
        assert demand >= 0
        gravity_tm[int(src), int(dst)] = demand

    simulate_all_traffic = np.zeros((net.get_num_nodes, net.get_num_nodes), dtype=np.float32)
    for src in range(net.get_num_nodes):
        for dst in range(net.get_num_nodes):
            if src == dst:
                continue
            simulate_all_traffic += oblivious_matrices[src][dst] * gravity_tm[src][dst]

    return oblivious_matrices, simulate_all_traffic


def print_link_loads(edges_traffic_list, title: str):
    print(title)
    print("-" * 50)
    for index, (e_src, e_dst, cap, load) in enumerate(edges_traffic_list):
        print("{}: Link_Source: {}; Link Destination: {}; Capacity: {} [MB] ;Load: {} [MB]".format(index + 1, e_src,
                                                                                                   e_dst,
                                                                                                   cap, load))

    print("=" * 100)


def rl_parsing(learning_path, network: NetworkClass):
    learning_path_file = open(learning_path, "rb")
    learning_matrices = np.load(learning_path_file)
    learning_matrices_sum = np.sum(learning_matrices, axis=0)

    gravity_traffic = net.gravity_traffic_map()
    gravity_tm = np.zeros((net.get_num_nodes, net.get_num_nodes), dtype=np.float32)
    for src, dst, demand in gravity_traffic:
        assert demand >= 0
        gravity_tm[int(src), int(dst)] = demand

    simulate_all_traffic = np.zeros((net.get_num_nodes, net.get_num_nodes), dtype=np.float32)
    for src in range(net.get_num_nodes):
        for dst in range(net.get_num_nodes):
            if src == dst:
                continue
            simulate_all_traffic += learning_matrices_sum[src][dst] * gravity_tm[src][dst]

    return learning_matrices, simulate_all_traffic


def routing_schemes_analyzing(simulate_all_traffic, net: NetworkClass):
    edges_traffic_list = list()
    for e_src in range(net.get_num_nodes):
        for e_dst in range(net.get_num_nodes):
            if (e_src, e_dst) not in net.edges:
                continue
            cap = net.get_edge_key((e_src, e_dst), EdgeConsts.CAPACITY_STR)
            edges_traffic_list.append((e_src, e_dst, cap, simulate_all_traffic[e_src][e_dst]))

    edges_traffic_list.sort(reverse=True, key=lambda e: e[3])

    return edges_traffic_list


if __name__ == "__main__":
    args = _getOptions()
    oblivious_path = args.oblivious_path
    learning_path = args.learning_path
    topology_url = args.topology_url
    net = NetworkClass(topology_zoo_loader(topology_url)).get_g_directed

    oblivious_matrices, simulate_all_traffic = oblivious_parsing(oblivious_path, net)
    edges_traffic_list = routing_schemes_analyzing(simulate_all_traffic, net)
    print_link_loads(edges_traffic_list, "Oblivious Links Loads")

    learning_matrices, simulate_all_traffic = rl_parsing(learning_path, net)
    edges_traffic_list = routing_schemes_analyzing(simulate_all_traffic, net)
    print_link_loads(edges_traffic_list, "Reinforcement Learning Loads")
