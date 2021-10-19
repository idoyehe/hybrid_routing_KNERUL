import json
from common.logger import logger
from Smart_Nodes_Routing.rl_env.model_functions import *
from common.RL_Envs.rl_env_consts import HyperparamertsStrings
import numpy as np

def run_tuning(hyperparameters: dict, config_folder: str):
    config_path = config_folder + "config.json"
    json_file = open(config_path, 'r')
    config = json.load(json_file)["tuning"]
    json_file.close()

    train_file = config_folder + config["train_file"]
    test_file = config_folder + config["test_file"]
    num_train_observations = config["num_train_observations"]
    num_test_observations = config["num_test_observations"]
    policy_updates = config["policy_updates"]

    weights_factor = hyperparameters[HyperparamertsStrings.WEIGHTS_FACTOR]
    weight_lb = hyperparameters[HyperparamertsStrings.WEIGHT_LB]
    weight_ub = hyperparameters[HyperparamertsStrings.WEIGHT_UB]

    learning_rate = hyperparameters[HyperparamertsStrings.LEARNING_RATE]
    batch_size = hyperparameters[HyperparamertsStrings.BATCH_SIZE]
    n_steps = hyperparameters[HyperparamertsStrings.N_STEPS]

    _envs = build_clean_smart_nodes_env(train_file, test_file, num_train_observations, num_test_observations, weights_factor=weights_factor,
                                        action_weight_lb=weight_lb, action_weight_ub=weight_ub)

    model = build_clean_smart_nodes_model(_envs, learning_rate, n_steps, batch_size)

    logger.info("Run configuration: {}".format(hyperparameters))

    try:
        model.learn(policy_updates * n_steps)
    except AssertionError:
        return -1e6

    single_env = _envs.envs[0].env

    return float(run_testing(model, single_env, num_test_observations))
