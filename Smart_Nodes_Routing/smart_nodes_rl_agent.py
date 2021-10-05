from common.logger import logger
from rl_env.model_functions import model_learn, greedy_best_smart_nodes_and_spr, model_continue_learning, run_testing
from argparse import ArgumentParser
from sys import argv


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-cfg", "--config_folder", type=str, help="The path of configuration folderv")
    parser.add_argument("-n_sn", "--number_smart_nodes", type=int, help="Number of smart nodes", default=1)
    parser.add_argument("-s_nodes", "--smart_nodes_set", type=eval, help="Smart Node set to examine", default=None)
    parser.add_argument("-l_agent", "--load_agent", type=str, help="Load a dumped agent", default=None)
    parser.add_argument("-l_net", "--load_network", type=str, help="Load a dumped Network object", default=None)
    parser.add_argument("-n_iter", "--number_of_iterations", type=int, help="Number of iteration", default=2)
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    args = _getOptions()
    config_folder = args.config_folder
    number_smart_nodes = args.number_smart_nodes
    smart_nodes_set = args.smart_nodes_set
    load_agent = args.load_agent
    load_network = args.load_network
    num_of_iterations = args.number_of_iterations

    if load_agent is None:
        logger.info("********* Iteration 0 Starts *********")
        model, single_env = model_learn(config_folder, "Iteration_0", load_agent, load_network, policy_updates=1000)
    else:
        model, single_env = model_learn(config_folder, "Iteration_0", load_agent, load_network, policy_updates=200)

        logger.info("***************** Iteration 0 Finished ******************")

    current_smart_nodes = tuple()
    for i in range(1, num_of_iterations + 1):
        logger.info("**** Iteration {}, Evaluating Smart Node  *****".format(i))
        link_weights, _ = model.predict(single_env.reset(), deterministic=True)
        link_weights = single_env.update_link_weights(link_weights)
        destination_based_sprs = single_env.get_optimizer.calculating_destination_based_spr(link_weights)
        env_train_observations = single_env.get_train_observations

        best_smart_nodes = greedy_best_smart_nodes_and_spr(single_env.get_network, env_train_observations[0], destination_based_sprs, number_smart_nodes, smart_nodes_set)

        current_smart_nodes = best_smart_nodes[0]
        single_env.set_network_smart_nodes_and_spr(current_smart_nodes, best_smart_nodes[2])

        logger.info("********** Iteration {}, Smart Nodes:{}  ***********".format(i, current_smart_nodes))
        logger.info("********* Iteration {} Starts, Agent is learning *********".format(i))
        learning_title = "Iteration {}".format(i)
        model, single_env = model_continue_learning(model, single_env, learning_title, policy_updates=300)

    logger.info("========================== Learning Process is Done =================================")

    run_testing(model, single_env, single_env.get_num_test_observations)
