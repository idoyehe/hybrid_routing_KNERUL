from common.network_class import *
from common.topologies import store_graph
from argparse import ArgumentParser
from sys import argv


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")

    parser.add_argument("-n", "--number_of_nodes", type=int, help="Number of nodes")
    parser.add_argument("-seed", "--seed", type=int, help="Random Seed")
    parser.add_argument("-alpha", "--alpha", type=float, help="alpha", default=0.15)
    parser.add_argument("-beta", "--beta", type=float, help="beta", default=0.15)
    parser.add_argument("-gamma", "--gamma", type=float, help="gamma", default=0.7)
    parser.add_argument("-delta_in", "--delta_in", type=float, help="delta_in", default=0.2)
    parser.add_argument("-delta_out", "--delta_out", type=float, help="delta_out", default=0.2)
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    args = _getOptions()
    n = args.number_of_nodes
    seed = args.seed
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    delta_in = args.delta_in
    delta_out = args.delta_out

    G = nx.generators.scale_free_graph(n, seed=seed, alpha=alpha, beta=beta, gamma=gamma)
    for edge in G.edges:
        G.edges[edge]["LinkSpeedRaw"] = 10000000000

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    net = NetworkClass(G)
    print("Number of Edges: {}".format(net.get_num_edges))
    net.print_network()
    G.name = "ScaleFree{}Nodes".format(n)
    store_graph(G, "scale_free_{}_nodes_{}_seed".format(n, seed))
