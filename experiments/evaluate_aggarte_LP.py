# import time
#
# from soft_min_smart_node_LP_optimizer import SoftMinSmartNodesOptimizer
# from argparse import ArgumentParser
# from sys import argv
# from common.utils import load_dump_file
# from common.network_class import NetworkClass
# from common.topologies import topology_zoo_loader
# import random
# import numpy as np
#
#
# def _getOptions(args=argv[1:]):
#     parser = ArgumentParser(description="Parses TMs Generating script arguments")
#
#     parser.add_argument("-p", "--dumped_path", type=str, help="The path of the dumped file")
#     parser.add_argument("-n", "--number_of_evaluations", type=int, help="Number of TMs evaluations")
#     options = parser.parse_args(args)
#     return options
#
#
# if __name__ == "__main__":
#     args = _getOptions()
#     dumped_path = args.dumped_path
#     n = args.number_of_evaluations
#     loaded_dict = load_dump_file(file_name=dumped_path)
#     tm_list = loaded_dict["tms"]
#     network = NetworkClass(topology_zoo_loader(url=loaded_dict["url"]))
#     optimizer = SoftMinSmartNodesOptimizer(network, False)
#     tm_evaluations = list()
#     link_weights = list()
#
#     for _ in range(n):
#         _tm = random.choice(tm_list)
#         tm_evaluations.append(_tm[0])
#         link_weights.append(np.random.rand(network.get_num_edges) * 2)
#
#     serial_time = time.time()
#     for i in range(n):
#         optimizer.step(link_weights[i], tm_evaluations[i], None)
#     print("Serial Time: {}".format(time.time() - serial_time))
#
#     all_at_once_time = time.time()
#     optimizer.step_tms(link_weights, tm_evaluations)
#     print("All at once Time: {}".format(time.time() - all_at_once_time))
#
