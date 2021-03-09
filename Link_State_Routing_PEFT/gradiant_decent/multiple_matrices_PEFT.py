from argparse import ArgumentParser
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_solver
from common.consts import EdgeConsts
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file, error_bound, extract_flows
from common.logger import *
from sys import argv
from random import shuffle


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    parser.add_argument("-n", "--number_of_matrices", type=int, help="The number of matrices to solve for")
    parser.add_argument("-p_l", "--probabilities_list", type=list, help="List of probabilities fro each matrix")
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    number_of_matrices = options.number_of_matrices
    probabilities_list = options.probabilities_list

    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))

    shuffle(loaded_dict["tms"])
