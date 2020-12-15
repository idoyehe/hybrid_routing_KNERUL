import numpy as np
from argparse import ArgumentParser
from sys import argv
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.consts import EdgeConsts


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for console output file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for rl vs oblivious file")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from")
    options = parser.parse_args(args)
    return options


def load_from_npc(file_path):
    matrix_file = open(file_path, "rb")
    matrix = np.load(matrix_file)
    matrix_file.close()
    obliv_most_congested = np.array([eval(e) for e in matrix[:, 0]]).reshape((matrix.shape[0], 2))
    rl_most_congested = np.array([eval(e) for e in matrix[:, 1]]).reshape((matrix.shape[0], 2))
    agrees = np.array([e == 'True' for e in matrix[:, 2]], dtype=bool).reshape((matrix.shape[0], 1))
    deltas = np.array([float(e) for e in matrix[:, 3]]).reshape((matrix.shape[0], 1))
    return np.concatenate((obliv_most_congested, rl_most_congested, agrees, deltas), axis=1)


def analyze_disagreements(net: NetworkClass, data_matrix):
    edges_matrix_disagreements = np.zeros(
        shape=(net.get_num_nodes, net.get_num_nodes, net.get_num_nodes, net.get_num_nodes), dtype=int)
    edges_matrix_deltas = np.zeros(
        shape=(net.get_num_nodes, net.get_num_nodes, net.get_num_nodes, net.get_num_nodes),
        dtype=np.float64)

    most_common_mistake = None
    most_common_mistake_counter = 0

    less_common_mistake = None
    less_common_mistake_counter = np.inf
    for row in data_matrix[np.where(data_matrix[:, 4] == 0)]:
        obliv_link = (int(row[0]), int(row[1]))
        rl_link = (int(row[2]), int(row[3]))

        assert obliv_link != rl_link
        delta = row[5]
        assert delta >= 0

        edges_matrix_disagreements[obliv_link + rl_link] += 1
        edges_matrix_deltas[obliv_link + rl_link] += delta
        if edges_matrix_disagreements[obliv_link + rl_link] > most_common_mistake_counter:
            most_common_mistake = (obliv_link, rl_link)
            most_common_mistake_counter = edges_matrix_disagreements[obliv_link + rl_link]

        if edges_matrix_disagreements[obliv_link + rl_link] <= less_common_mistake_counter:
            less_common_mistake = (obliv_link, rl_link)
            less_common_mistake_counter = edges_matrix_disagreements[obliv_link + rl_link]

    assert most_common_mistake_counter == np.max(edges_matrix_disagreements)
    for d1 in range(net.get_num_nodes):
        for d2 in range(net.get_num_nodes):
            for d3 in range(net.get_num_nodes):
                for d4 in range(net.get_num_nodes):
                    if edges_matrix_disagreements[d1, d2, d3, d4] <= 0:
                        continue
                    obliv_link = (d1, d2)
                    rl_link = (d3, d4)
                    edges_matrix_deltas[obliv_link + rl_link] /= edges_matrix_disagreements[obliv_link + rl_link]

    print("most common mistake: {} <--> {}".format(most_common_mistake[0], most_common_mistake[1]))
    print("most common mistake congestion ratio delta: {}".format(edges_matrix_deltas[sum(most_common_mistake, ())]))

    print("less common mistake: {} <--> {}".format(less_common_mistake[0], less_common_mistake[1]))
    print("less common mistake congestion ratio delta: {}".format(edges_matrix_deltas[sum(less_common_mistake, ())]))

    return edges_matrix_disagreements, edges_matrix_deltas


def analysis_deltas(edges_matrix_disagreements, edges_matrix_deltas):
    mean_congestion_ratio_delta = np.mean(edges_matrix_deltas[np.where(edges_matrix_disagreements > 0)])
    std_congestion_ratio_delta = np.std(edges_matrix_deltas[np.where(edges_matrix_disagreements > 0)])
    max_congestion_ratio_delta = np.max(edges_matrix_deltas[np.where(edges_matrix_disagreements > 0)])

    print("mean_congestion_ratio_delta: {}".format(mean_congestion_ratio_delta))
    print("std_congestion_ratio_delta: {}".format(std_congestion_ratio_delta))
    print("max_congestion_ratio_delta: {}".format(max_congestion_ratio_delta))


def analysis_star(edges_matrix_disagreements, edges_matrix_deltas):
    print("\n===============\n")
    print("Analysis Stars")
    print("--------------")

    print("when RL choose 11->5 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 5])))
    print("when Oblivious choose 11->5 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 5, :, :])))
    print("--------------")

    print("when RL choose 11->6 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 6])))
    print("when Oblivious choose 11->6 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 6, :, :])))
    print("--------------")

    print("when RL choose 11->7 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 6])))
    print("when Oblivious choose 11->7 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 6, :, :])))
    print("--------------")

    print("when RL choose 11->8 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 8])))
    print("when Oblivious choose 11->8 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 8, :, :])))
    print("--------------")

    print("when RL choose 11->4 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 4])))
    print("when Oblivious choose 11->4 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 4, :, :])))
    print("--------------")

    print("when RL choose 3->1 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 0, 1])))
    print("when Oblivious choose 3->1 disagreements: {}".format(np.sum(edges_matrix_disagreements[0, 1, :, :])))
    print("--------------")

    print("when RL choose 3->0 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 0, 3])))
    print("when Oblivious choose 3->0 disagreements: {}".format(np.sum(edges_matrix_disagreements[0, 3, :, :])))
    print("--------------")

    print("when RL choose 10->9 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 10, 9])))
    print("when Oblivious choose 10->9  disagreements: {}".format(np.sum(edges_matrix_disagreements[10, 9, :, :])))


def analysis_middle(edges_matrix_disagreements, edges_matrix_deltas):
    print("\n===============\n")
    print("Analysis Middle")
    print("--------------")

    print("when RL choose 11->10 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 10])))
    print("when Oblivious choose 11->10 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 10, :, :])))
    print("--------------")

    print("when RL choose 10->11 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 10, 11])))
    print("when Oblivious choose 10->11 disagreements: {}".format(np.sum(edges_matrix_disagreements[10, 11, :, :])))
    print("--------------")

    print("when RL choose 10->3 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 10, 3])))
    print("when Oblivious choose 10->3 disagreements: {}".format(np.sum(edges_matrix_disagreements[10, 3, :, :])))
    print("--------------")

    print("when RL choose 11->2 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 2])))
    print("when Oblivious choose 11->2 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 2, :, :])))
    print("--------------")

    print("when RL choose 2->11 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 2, 11])))
    print("when Oblivious choose 2->11 disagreements: {}".format(np.sum(edges_matrix_disagreements[2, 11, :, :])))
    print("--------------")

    print("when RL choose 2->3 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 2, 3])))
    print("when Oblivious choose 2->3 disagreements: {}".format(np.sum(edges_matrix_disagreements[2, 3, :, :])))
    print("--------------")

    print("when RL choose 11->3 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 11, 3])))
    print("when Oblivious choose 11->3 disagreements: {}".format(np.sum(edges_matrix_disagreements[11, 3, :, :])))
    print("--------------")

    print("when RL choose 3->11 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 3, 11])))
    print("when Oblivious choose 3->11 disagreements: {}".format(np.sum(edges_matrix_disagreements[3, 11, :, :])))
    print("--------------")

    print("when RL choose 3->10 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 3, 10])))
    print("when Oblivious choose 3->10 disagreements: {}".format(np.sum(edges_matrix_disagreements[3, 10, :, :])))
    print("--------------")

    print("when RL choose 3->2 disagreements: {}".format(np.sum(edges_matrix_disagreements[:, :, 3, 2])))
    print("when Oblivious choose 3->2 disagreements: {}".format(np.sum(edges_matrix_disagreements[3, 2, :, :])))


if __name__ == "__main__":
    args = _getOptions()
    dumped_path = args.dumped_path

    topology_url = args.topology_url
    net = NetworkClass(topology_zoo_loader(topology_url)).get_g_directed

    data_matrix = load_from_npc(dumped_path)
    edges_matrix_disagreements, edges_matrix_deltas = analyze_disagreements(net, data_matrix)
    analysis_deltas(edges_matrix_disagreements, edges_matrix_deltas)
    analysis_star(edges_matrix_disagreements, edges_matrix_deltas)
    analysis_middle(edges_matrix_disagreements, edges_matrix_deltas)
