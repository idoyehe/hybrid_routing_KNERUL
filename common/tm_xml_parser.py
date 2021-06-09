from sys import argv
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from common.static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver
from common.static_routing.oblivious_routing import *
from common.topologies import *
from common.network_class import *
import numpy as np
import os

NODES_DICT = {"ATLA-M5": 0,
              "ATLAng": 1,
              "CHINng": 2,
              "DNVRng": 3,
              "HSTNng": 4,
              "IPLSng": 5,
              "KSCYng": 6,
              "LOSAng": 7,
              "NYCMng": 8,
              "SNVAng": 9,
              "STTLng": 10,
              "WASHng": 11}


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-topo", "--topology_url", type=str, help="The url to load graph topology from",
                        default="C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\common\\graphs_gmls\\Abilene_07.txt")
    parser.add_argument("-dir", "--directory", type=str, help="The directory to fetch files from")
    options = parser.parse_args(args)
    return options


def __parsing_single_tm_xml_file(net, file_path):
    xml_tree = ET.parse(file_path)
    tm = np.zeros(shape=(net.get_num_nodes, net.get_num_nodes), dtype=np.float64)
    for tmf in xml_tree.iter("TrafficMatrixFile"):
        for intm in tmf.iter('IntraTM'):
            for src_obj in intm.iter('src'):
                src = NODES_DICT[src_obj.attrib["id"]]
                for dst_object in src_obj.iter('dst'):
                    dst = NODES_DICT[dst_object.attrib["id"]]
                    if src == dst:
                        continue
                    demand = np.float64(dst_object.text)
                    tm[src, dst] = demand
    return tm


def __parsing_single_directory(net, dir_path):
    tm_list = list()
    for filename in os.listdir(dir_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(dir_path, filename)
            tm_list.append(__parsing_single_tm_xml_file(net, file_path))
        else:
            continue
    return tm_list


def _generate_traffic_matrix_baseline(net: NetworkClass, tm_list, oblivious_routing_per_edge):
    tm_list_baseline = list()
    for index, tm in enumerate(tm_list):
        opt_ratio, _, _ = optimal_load_balancing_LP_solver(net, tm)
        obliv_ratio, _, _ = calculate_congestion_per_matrices(net=net, traffic_matrix_list=[(tm, opt_ratio)],
                                                              oblivious_routing_per_edge=oblivious_routing_per_edge)
        obliv_ratio = obliv_ratio[0]

        tm_list_baseline.append((tm, opt_ratio, obliv_ratio))
        logger.info("Current TM {} with optimal routing {}".format(index, opt_ratio))
    return tm_list_baseline


if __name__ == "__main__":
    args = _getOptions()
    dir_path = args.directory
    topology_url = args.topology_url
    net = NetworkClass(topology_zoo_loader(topology_url, units=SizeConsts.ONE_Kb))
    oblivious_ratio, oblivious_routing_per_edge, oblivious_routing_per_flow = oblivious_routing(net)
    print("The oblivious ratio for {} is {}".format(net.get_name, oblivious_ratio))



    tm_list = __parsing_single_directory(net, dir_path)
    tms = _generate_traffic_matrix_baseline(net, tm_list, oblivious_routing_per_edge)

    dict2dump = {
        "tms": tms,
        "url": topology_url,
        "oblivious_routing": {
            "per_edge": oblivious_routing_per_edge,
            "per_flow": oblivious_routing_per_flow
        },
        "tms_type": "database", }

    folder_name: str = os.getcwd() + "/TMs_DB/{}".format(net.get_name)
    file_name: str = os.getcwd() + "/TMs_DB/{}/{}_tms_{}X{}_length_{}_database".format(net.get_name,
                                                                                          net.get_name,
                                                                                          net.get_num_nodes,
                                                                                          net.get_num_nodes,
                                                                                          len(tms))

    os.makedirs(folder_name, exist_ok=True)
    dump_file = open(file_name, 'wb')
    pickle.dump(dict2dump, dump_file)
    dump_file.close()
