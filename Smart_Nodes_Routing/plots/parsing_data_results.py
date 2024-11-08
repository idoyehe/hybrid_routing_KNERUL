import os
import numpy as np
import json as js

DIR_NAME = os.path.dirname(__file__)
JSON_PATH = os.path.join(DIR_NAME, "../experiments/Evaluations_V3.json")


def __parse_training_set(data_training_set, obliv_base, x_labels):
    baseline = data_training_set["oblivious_routing"] if obliv_base else data_training_set["optimal"]
    oblivious_routing_ratio = (data_training_set["oblivious_routing"] / baseline - 1) * 100
    cope_routing_ratio = (data_training_set["cope_routing"] / baseline - 1) * 100
    averaged_tm_optimal_routing_ratio = (data_training_set["averaged_tm_optimal_routing_scheme"] / baseline - 1) * 100
    max_tm_optimal_routing_ratio = (data_training_set["max_tm_optimal_routing_scheme"] / baseline - 1) * 100
    weight_initialization_ratio = (data_training_set["smart_weight_initialization"] / baseline - 1) * 100

    y_data = list()
    for label in x_labels:
        label_ratios = [(s / baseline - 1) * 100 for s in data_training_set[label]]
        y_data.append((np.mean(label_ratios), np.std(label_ratios)))
    # y_data.insert(1, (weight_initialization_ratio, 0))

    return y_data, oblivious_routing_ratio, averaged_tm_optimal_routing_ratio, max_tm_optimal_routing_ratio, cope_routing_ratio


def __parse_testing_set(data_testing_set, obliv_base, x_labels, weight_init=False):
    baselines = data_testing_set["oblivious_routing"] if obliv_base else data_testing_set["optimal"]
    oblivious_routing_ratios = [(s / baselines[i] - 1) * 100 for i, s in enumerate(data_testing_set["oblivious_routing"])]
    cope_routing_ratios = [(s / baselines[i] - 1) * 100 for i, s in enumerate(data_testing_set["cope_routing"])]
    averaged_tm_optimal_routing_ratios = [(s / baselines[i] - 1) * 100 for i, s in enumerate(data_testing_set["averaged_tm_optimal_routing_scheme"])]
    max_tm_optimal_routing_ratios = [(s / baselines[i] - 1) * 100 for i, s in enumerate(data_testing_set["max_tm_optimal_routing_scheme"])]
    if weight_init:
        weight_initialization_ratios = [(s / baselines[i] - 1) * 100 for i, s in enumerate(data_testing_set["smart_weight_initialization"])]

    y_data = list()
    for label in x_labels:
        label_ratios = [(s / baselines[i] - 1) * 100 for i, s in enumerate(data_testing_set[label])]
        y_data.append((np.mean(label_ratios), np.std(label_ratios)))

    if weight_init:
        y_data.insert(1, (np.mean(weight_initialization_ratios), np.std(weight_initialization_ratios)))

    return y_data, oblivious_routing_ratios, averaged_tm_optimal_routing_ratios, max_tm_optimal_routing_ratios, cope_routing_ratios


def parse_rl_optimization_cross_topologies(traffic):
    # Opening JSON file
    f = open(JSON_PATH)
    # returns JSON object as a dictionary
    data = js.load(f)
    f.close()
    raw_labels = ("naive_RL", "first_RL_phase")
    topologies_raw_names = ("claranet", "goodNet", "scaleFree30Nodes", "GEANT", "china_telecom")
    topologies = ("Claranet", "GoodNet", "Scale Free\n30 Nodes", "GEANT", "China\nTelecom")

    y_data = dict()
    x_labels = ("Random Initialized RL", "Link Weight Initialization", "Link Weight Initialized RL")

    for i, label in enumerate(x_labels):
        y_data[label] = []
        for topo in topologies_raw_names:
            _y_data, _, _, _, _ = __parse_testing_set(data[topo][traffic]["results"]["testing_sets"], False, raw_labels, True)
            y_data[label].append(_y_data[i])

    return topologies, y_data


def parsing_data_results(method_name, topology_name, traffic, obliv_base):
    # Opening JSON file
    f = open(JSON_PATH)
    # returns JSON object as a dictionary
    data = js.load(f)
    f.close()
    data_topo_traffic = data[topology_name][traffic]['results']
    x_raw_labels = ["naive_RL", "first_RL_phase", "1_key_node", "2_key_node", "3_key_node", "4_key_node"]
    if "5_key_node" in data_topo_traffic["RL_training_set"].keys():
        x_raw_labels.append("5_key_node")
    if "all_key_node" in data_topo_traffic["RL_training_set"].keys():
        x_raw_labels.append("all_key_node")

    y_data_rl_train, oblivious_routing_ratio_rl_train, averaged_tm_optimal_routing_ratio_rl_train, max_tmoptimal_routing_ratio_rl_train, cope_routing_ratios_rl_train = __parse_training_set(
        data_topo_traffic["RL_training_set"], obliv_base, x_raw_labels)

    y_data_lp_train, oblivious_routing_ratio_lp_train, averaged_tm_optimal_routing_ratio_lp_train, max_tm_optimal_routing_ratio_lp_train, cope_routing_ratios_lp_train = __parse_training_set(
        data_topo_traffic["LP_training_set"], obliv_base, x_raw_labels)

    y_data_test, oblivious_routing_ratios_test, averaged_tm_optimal_routing_ratios_test, max_tm_optimal_routing_ratios_test, cope_routing_ratios_test = __parse_testing_set(
        data_topo_traffic["testing_sets"], obliv_base, x_raw_labels)

    oblivious_routing_ratios = [oblivious_routing_ratio_rl_train, oblivious_routing_ratio_lp_train] + oblivious_routing_ratios_test
    cope_routing_ratios = [cope_routing_ratios_rl_train, cope_routing_ratios_lp_train] + cope_routing_ratios_test

    averaged_tm_optimal_routing_ratios = [averaged_tm_optimal_routing_ratio_rl_train,
                                          averaged_tm_optimal_routing_ratio_lp_train] + averaged_tm_optimal_routing_ratios_test

    max_tm_optimal_routing_ratios = [max_tmoptimal_routing_ratio_rl_train,
                                     max_tm_optimal_routing_ratio_lp_train] + max_tm_optimal_routing_ratios_test

    if obliv_base:
        h_lines = [("Averaged TM Optimal Routing", "black", "dotted", np.mean(averaged_tm_optimal_routing_ratios)),
                   ("Max TM Optimal Routing", "red", "dashed", np.mean(max_tm_optimal_routing_ratios))]
    else:
        h_lines = [("Oblivious Routing", "blue", "solid", np.mean(oblivious_routing_ratios)),
                   ("Cope Routing", "orange", "solid", np.mean(cope_routing_ratios)),
                   ("Averaged TM Optimal Routing", "black", "dotted", np.mean(averaged_tm_optimal_routing_ratios)),
                   ("Max TM Optimal Routing", "red", "dashed", np.mean(max_tm_optimal_routing_ratios))
                   ]

    y_data = dict()
    y_data["Non-Key Nodes Train Set"] = y_data_rl_train
    y_data["Key Nodes Train Set"] = y_data_lp_train
    y_data["Test Sets"] = y_data_test

    x_labels = ["Random\nInitialized\nRL",
                f"{method_name}\nRL",
                f"{method_name}\n1 Key\nNode",
                f"{method_name}\n2 Key\nNodes",
                f"{method_name}\n3 Key\nNodes",
                f"{method_name}\n4 Key\nNodes"]

    if "5_key_node" in data_topo_traffic["RL_training_set"].keys():
        x_labels.append(f"{method_name}\n5 Key\nNodes")
    if "all_key_node" in data_topo_traffic["RL_training_set"].keys():
        x_labels.append(f"{method_name}\nAll Key\nNodes")

    return x_labels, y_data, h_lines
