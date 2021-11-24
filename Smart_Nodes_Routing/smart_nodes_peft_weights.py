from common.logger import logger
from rl_env.model_functions import greedy_best_smart_nodes_and_spr, run_testing, get_json_file_from_cfg, model_learn, model_continue_learning, \
    load_network_and_update_env
from argparse import ArgumentParser
from sys import argv
from common.utils import load_dump_file


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-cfg", "--config_folder", type=str, help="The path of configuration folderv")
    parser.add_argument("-n_sn", "--number_smart_nodes", type=int, help="Number of smart nodes", default=1)
    parser.add_argument("-s_nodes", "--smart_nodes_set", type=eval, help="Smart Node set to examine", default=None)
    parser.add_argument("-l_net", "--load_network", type=str, help="Load a dumped Network object", default=None)
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    args = _getOptions()
    config_folder = args.config_folder
    number_smart_nodes = args.number_smart_nodes
    smart_nodes_set = args.smart_nodes_set
    load_network = args.load_network
    config = get_json_file_from_cfg(config_folder)

    lp_file = config_folder + config["lp_file"]
    lp_tms = load_dump_file(lp_file)["tms"]
    train_file = config_folder + config["train_file"]
    train_init_weight = load_dump_file(train_file)["initial_weights"]
    test_file = config_folder + config["test_file"]
    test_tms = load_dump_file(test_file)["tms"]

    logger.info("********** Build Destination Based Splitting Ratios  ***********")

    link_weights = train_init_weight
    model, single_env, _ = model_learn(config_folder, "smart_0_nodes_net")
    if load_network is not None:
        net, _ = load_network_and_update_env(load_network, single_env)
    else:
        destination_based_sprs = single_env.get_optimizer.calculating_destination_based_spr(link_weights)
        best_smart_nodes = greedy_best_smart_nodes_and_spr(single_env.get_network, lp_tms, destination_based_sprs, number_smart_nodes,
                                                           smart_nodes_set)

        current_smart_nodes = best_smart_nodes[0]
        single_env.set_network_smart_nodes_and_spr(current_smart_nodes, best_smart_nodes[2])
        logger.info("********** Smart Nodes:{}  ***********".format(current_smart_nodes))
        model, single_env = model_continue_learning(model, single_env, "smart_{}_nodes_net".format(number_smart_nodes), policy_updates=0)

        logger.info("========================== Evaluating Smart Nodes Process is Done =================================")

    run_testing(model, single_env, single_env.get_num_test_observations, link_weights)
