import numpy as np
from argparse import ArgumentParser
from sys import argv
import matplotlib.pyplot as plt


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for console output file")
    parser.add_argument("-p", "--file_path", type=str, help="The path for console output file")
    parser.add_argument("-n_links", "--number_of_links", type=int, help="Number of links to print")
    parser.add_argument("-s_t", "--start_time", type=int, help="Start time to show", default=0)
    options = parser.parse_args(args)
    return options


# def load_numpy_object_from_file(file_path: str):
#     link_weights_file = open(file_path, "rb")
#     cumulative_list = list()
#
#     while True:
#         try:
#             cumulative_list.append(np.load(link_weights_file))
#         except Exception as e:
#             print("Read from file done, EOF")
#             link_weights_file.close()
#             break
#     assert len(cumulative_list) > 0
#     link_weights_matrix = np.array(cumulative_list).transpose()
#     link_weights_file = open(file_path, "wb")
#     np.save(link_weights_file, link_weights_matrix)
#     link_weights_file.close()

def load_numpy_object_from_file(file_path: str):
    link_weights_file = open(file_path, "rb")
    link_weights_matrix = np.load(link_weights_file)
    link_weights_file.close()
    assert len(link_weights_matrix) > 0
    return link_weights_matrix


def plot_link_weights(link_weights_matrix, number_of_links: int, start_time):
    for link_index in range(number_of_links):
        fig = plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
        link_weights_data = link_weights_matrix[link_index][start_time:]
        plt.title("Link No. {} STD is {}".format(link_index, np.std(link_weights_data)))
        plt.xlabel("# Timesteps")
        plt.ylabel("Link Weight")
        plt.plot(np.arange(start_time, start_time + len(link_weights_data), step=1), link_weights_data)
        plt.show()


if __name__ == "__main__":
    file_path = _getOptions().file_path
    number_of_links = _getOptions().number_of_links
    start_time = _getOptions().start_time
    print("Loading from file path: {}".format(file_path))
    link_weights_matrix = load_numpy_object_from_file(file_path)
    print("Done! loading from file")
    plot_link_weights(link_weights_matrix, number_of_links, start_time)
