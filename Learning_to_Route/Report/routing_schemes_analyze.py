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


def load_from_npc(file_path):
    matrix_file = open(file_path, "rb")
    total_archs_load = np.load(matrix_file)
    matrix_file.close()
    return total_archs_load


def print_link_loads(edges_traffic_list, title: str):
    print(title)
    print("-" * 50)
    for index, (e_src, e_dst, cap, load, tcg) in enumerate(edges_traffic_list):
        print(
            "{}: Link_Source: {}; Link Destination: {}; Capacity: {} [MB] ;Load: {} [MB]; Total Congestion: {:.2f}".format(
                index + 1, e_src,
                e_dst,
                cap, load, tcg))

    print("=" * 100)


def routing_records_analyzing(simulate_all_traffic, net: NetworkClass):
    edges_traffic_list = list()
    for e_src, e_dst in net.edges:
        cap = net.get_edge_key((e_src, e_dst), EdgeConsts.CAPACITY_STR)
        total_congestion = simulate_all_traffic[e_src][e_dst] / cap
        edges_traffic_list.append((e_src, e_dst, cap, simulate_all_traffic[e_src][e_dst], total_congestion))

    edges_traffic_list.sort(reverse=True, key=lambda e: e[4])

    return edges_traffic_list


if __name__ == "__main__":
    args = _getOptions()
    oblivious_path = args.oblivious_path
    learning_path = args.learning_path
    topology_url = args.topology_url
    net = NetworkClass(topology_zoo_loader(topology_url)).get_g_directed

    all_traffic_oblivous = load_from_npc(oblivious_path)
    edges_traffic_list = routing_records_analyzing(all_traffic_oblivous, net)
    print_link_loads(edges_traffic_list, "Oblivious Links Loads")

    all_traffic_rl = load_from_npc(learning_path)
    temp = np.zeros_like(all_traffic_oblivous)

    for idx, load in enumerate(all_traffic_rl):
        link = net.get_id2edge()[idx]
        temp[link] = load

    all_traffic_rl = temp
    edges_traffic_list = routing_records_analyzing(all_traffic_rl, net)
    print_link_loads(edges_traffic_list, "Reinforcement Learning Loads")
