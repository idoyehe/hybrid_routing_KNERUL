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


def load_numpy_object_from_file(file_path: str):
    link_weights_file = open(file_path, "rb")
    cumulative_list = list()

    while True:
        try:
            cumulative_list.append(np.load(link_weights_file))
        except Exception as e:
            print("Read from file done, EOF")
            link_weights_file.close()
            break
    assert len(cumulative_list) > 0
    link_weights_matrix = np.array(cumulative_list).transpose()
    link_weights_file = open(file_path, "wb")
    np.save(link_weights_file, link_weights_matrix)
    link_weights_file.close()


if __name__ == "__main__":
    file_path = _getOptions().file_path
    number_of_links = _getOptions().number_of_links
    start_time = _getOptions().start_time
    print("Loading from file path: {}".format(file_path))
    link_weights_matrix = load_numpy_object_from_file(file_path)
    print("Done! loading from file")
