from network_class import *
from argparse import ArgumentParser
from sys import argv
from common.utils import load_dump_file
from topologies import topology_zoo_loader


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


def get_min_degree(net: NetworkClass):
    min_node = np.argmin(net.get_degrees())
    min_deg = np.min(net.get_degrees())
    return min_node, min_deg


def get_max_degree(net: NetworkClass):
    max_node = np.argmax(net.get_degrees())
    max_deg = np.max(net.get_degrees())
    return max_node, max_deg


def get_averaged_degree(net: NetworkClass):
    return np.mean(net.get_degrees())


def get_averaged_capacity(net: NetworkClass):
    return np.mean(net.get_edges_capacities())


def traffic_analyze(tms):
    _mean = list()
    _var = list()
    max_demand = 0
    min_demand = np.inf
    for t in tms:
        t = t[0].flatten()
        _mean.append(np.mean(t))
        _var.append(np.var(t))
        max_demand = max(max_demand, t.max())
        min_demand = min(min_demand, t[np.where(t > 0)].min())

    return [np.mean(_mean), np.mean(_var), max_demand, min_demand]


if __name__ == "__main__":
    args = _getOptions()
    dumped_path = args.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"])).get_g_directed
    print("Topology Name: {}".format(net.get_name))
    print("Number of Nodes: {}".format(net.get_num_nodes))
    print("Number of Edges: {}".format(net.get_num_edges))
    print("Maximum of degree: {}".format(get_max_degree(net)))
    print("Minimum of degree: {}".format(get_min_degree(net)))
    print("Averaged degree: {}".format(get_averaged_degree(net)))
    print("Averaged capacity: {}".format(get_averaged_capacity(net)))
    print(
        "Mean of Traffic demands: {}\nVariance of Traffic demands: {}\nMaximum of Traffic demands: {}\nMinimum of Traffic demands: {}".format(
            *traffic_analyze(loaded_dict["tms"])))
    net.print_network()
