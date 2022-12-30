import typing

import numpy as np
import xmltodict
import os
from network_class import NetworkClass
from common.topologies import *
from common.topologies import *
import re, pickle
from common.utils import load_dump_file
from common.logger import logger
from static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver
from common.consts import TMType, DumpsConsts

label2id = {
    "se1": 0,
    "nl1": 1,
    "be1": 2,
    "hr1": 3,
    "ie1": 4,
    "ny1": 5,
    "pt1": 6,
    "ch1": 7,
    "si1": 8,
    "it1": 9,
    "pl1": 10,
    "fr1": 11,
    "uk1": 12,
    "lu1": 13,
    "at1": 14,
    "il1": 15,
    "cz1": 16,
    "gr1": 17,
    "hu1": 18,
    "de1": 19,
    "es1": 20,
    "sk1": 21
}

folder = "C:\\Users\\idoye\\PycharmProjects\\hybrid_routing_with_KNERL\\common\\TMs_DB\\GEANT_V2\\XMLs"
new_folder = "C:\\Users\\idoye\\PycharmProjects\\hybrid_routing_with_KNERL\\common\\TMs_DB\\GEANT_V2"


def one_sample_tm_base():
    for index, filename in enumerate(sorted(os.listdir(folder))):
        nodes = len(label2id.keys())
        with open(os.path.join(folder, filename), 'r') as xml_file:  # open in readonly mode
            # time = re.findall('\d{8}-\d{4}', xml_file.name)[0]
            # np_filename = os.path.join(new_folder, f'{time}.np')
            data_dict = xmltodict.parse(xml_file.read())
            network = data_dict.get('network', None)
            if network:
                network_demands = network.get('demands', None)
                if network_demands:
                    demands = network_demands.get('demand', None)
                    if demands:
                        tm = np.zeros(shape=(nodes, nodes))
                        for d in demands:
                            src = d['source'].split(".")[0]
                            dst = d['target'].split(".")[0]
                            val = float(d['demandValue'])
                            tm[label2id[src], label2id[dst]] = val  # * 1e-3
                        # with open(np_filename, 'wb') as dump_file:
                        #     pickle.dump({'time': time, 'tm': tm}, dump_file)
                        #     dump_file.close()
                        print(f"TM index: {index}")
                        yield tm


def generate_traffic_matrix_baseline(net: NetworkClass, total_matrices: int, tms_generator: typing.Generator):
    logger.info("Generating baseline of traffic matrices to evaluate of length {}".format(total_matrices))
    lb_expected_congestion = 0
    tms_opt_zipped_list = list()
    for index in range(total_matrices):
        tm = next(tms_generator)
        opt_congestion, _ = optimal_load_balancing_LP_solver(net, tm)
        lb_expected_congestion += opt_congestion
        tms_opt_zipped_list.append((tm, opt_congestion))
        logger.info("Current TM {} with optimal routing {}".format(index, opt_congestion))
    lb_expected_congestion /= total_matrices
    logger.info("Lower Bound Expected Congestion: {}".format(lb_expected_congestion))
    return tms_opt_zipped_list


if __name__ == "__main__":
    topology_url = "C:\\Users\\idoye\\PycharmProjects\\hybrid_routing_with_KNERL\\common\\graphs_gmls\\GEANT_V2.txt"

    for item in os.listdir(new_folder):
        if item.endswith(".np"):
            os.remove(item)

    net_direct = NetworkClass(topology_zoo_loader(topology_url))
    assert net_direct.get_num_nodes == len(label2id.keys())

    tms_generator = one_sample_tm_base()

    dump_file_names = [("GEANT_V2_tms_22X2_length_1024_real_traffic_train", 1024),
                       ("GEANT_V2_tms_22X2_length_4096_real_traffic_LP", 4096),
                       ("GEANT_V2_tms_22X2_length_1024_real_traffic_test_0", 1024),
                       ("GEANT_V2_tms_22X2_length_1024_real_traffic_test_1", 1024),
                       ("GEANT_V2_tms_22X2_length_1024_real_traffic_test_2", 1024),
                       ("GEANT_V2_tms_22X2_length_1024_real_traffic_test_3", 1024),
                       ("GEANT_V2_tms_22X2_length_1024_real_traffic_test_4", 1024)
                       ]

    for filename, total in dump_file_names:
        np_filename = os.path.join(new_folder, filename)
        tms_list = generate_traffic_matrix_baseline(net_direct, total, tms_generator)
        with open(np_filename, 'wb') as dump_file:
            pickle.dump({DumpsConsts.TMs: tms_list, DumpsConsts.MATRIX_TYPE: TMType.REAL}, dump_file)
            dump_file.close()
