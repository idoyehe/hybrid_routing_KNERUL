import numpy as np
from argparse import ArgumentParser
from sys import argv
import matplotlib.pyplot as plt
from topologies import topology_zoo_loader
from network_class import NetworkClass, nx


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for console output file")
    parser.add_argument("-p", "--file_path", type=str, help="The path for console output file")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from")
    parser.add_argument("-n_links", "--number_of_links", type=int, help="Number of links to print",default=-1)
    parser.add_argument("-s_t", "--start_time", type=int, help="Start time to show", default=0)
    parser.add_argument("-ex_p", "--export_path", type=str, help="path to export new graph GML", default="")
    options = parser.parse_args(args)
    return options


def load_numpy_object_from_file(file_path: str):
    link_weights_file = open(file_path, "rb")
    link_weights_matrix = np.load(link_weights_file)
    link_weights_file.close()
    assert len(link_weights_matrix) > 0
    return link_weights_matrix


def plot_link_weights(link_weights_matrix, link_map, number_of_links: int, start_time):
    links_weights_data_dict: dict = dict()
    for link_index in range(number_of_links):
        fig = plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
        link_weights_data = link_weights_matrix[link_index][start_time:]
        link_name = link_map[link_index]
        links_weights_data_dict[link_name] = {"mean": np.mean(link_weights_data),
                                              "std": np.std(link_weights_data),
                                              "var": np.var(link_weights_data)}
        plt.title("Link {} STD is {}".format(link_name, np.std(link_weights_data)))
        plt.xlabel("# Timesteps")
        plt.ylabel("Link Weight")
        plt.plot(np.arange(start_time, start_time + len(link_weights_data), step=1), link_weights_data)
        plt.show()
        plt.close()
    return links_weights_data_dict


def modify_network(net: NetworkClass, links_weights_data_dict: dict):
    for link, data in links_weights_data_dict.items():
        for key, value in data.items():
            net.edges[link][key] = str(value)
    return net


if __name__ == "__main__":
    args = _getOptions()
    file_path = args.file_path
    number_of_links = args.number_of_links
    start_time = args.start_time
    topology_url = args.topology_url
    export_path = args.export_path
    net = NetworkClass(topology_zoo_loader(topology_url))
    number_of_links = number_of_links if number_of_links > 0 else net.get_num_edges
    print("Loading from file path: {}".format(file_path))
    link_weights_matrix = load_numpy_object_from_file(file_path)
    print("Done! loading from file")
    links_weights_data_dict = plot_link_weights(link_weights_matrix, net.get_id2edge_map(), number_of_links, start_time)
    net = modify_network(net, links_weights_data_dict)
    nx.write_gml(net.get_graph, export_path)
