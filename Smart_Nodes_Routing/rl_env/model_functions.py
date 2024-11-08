import torch
from gym import envs, register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from common.logger import logger
from common.network_class import NetworkClass
from common.RL_Envs.rl_env_consts import *
from common.utils import find_nodes_subsets, SEPERATOR, DEVICE
from Smart_Nodes_Routing.rl_env.RL_smart_nodes import RL_Smart_Nodes
from Smart_Nodes_Routing.rl_env.smart_nodes_multiple_matrices_MCF_2 import matrices_mcf_LP_with_smart_nodes_solver
import numpy as np
import json
from functools import partial
from tabulate import tabulate


def _create_random_TMs_list(traffic_matrices_list):
    return np.array([t[0] for t in traffic_matrices_list])


def build_clean_smart_nodes_env(train_file: str,
                                test_file: str,
                                num_train_episodes: int,
                                num_test_episodes: int,
                                history_length: int = 0,
                                weights_factor=EnvConsts.WEIGHTS_FACTOR,
                                action_weight_lb=EnvConsts.WEIGHT_LB,
                                action_weight_ub=EnvConsts.WEIGHT_UB,
                                n_envs=2):
    logger.info("Train data loaded from: {}".format(train_file))
    logger.info("Test data loaded from: {}".format(test_file))

    logger.info("Train Episodes: {}".format(num_train_episodes))
    logger.info("Test Episodes: {}".format(num_test_episodes))

    if EnvsStrings.RL_ENV_SMART_NODES_GYM_ID in envs.registry.env_specs:
        del envs.registry.env_specs[EnvsStrings.RL_ENV_SMART_NODES_GYM_ID]

    register(id=EnvsStrings.RL_ENV_SMART_NODES_GYM_ID,
             entry_point='Smart_Nodes_Routing.rl_env.RL_smart_nodes:RL_Smart_Nodes',
             kwargs={
                 'history_length': history_length,
                 'path_dumped': train_file,
                 'test_file': test_file,
                 'num_train_episodes': num_train_episodes,
                 'num_test_episodes': num_test_episodes,
                 'weights_factor': weights_factor,
                 'action_weight_lb': action_weight_lb,
                 'action_weight_ub': action_weight_ub})

    return make_vec_env(EnvsStrings.RL_ENV_SMART_NODES_GYM_ID, n_envs=n_envs)


def build_clean_smart_nodes_model(model_envs, learning_rate: float, n_steps: int,
                                  batch_size: int,
                                  log_std_init: float,
                                  mlp_arch=None,
                                  gamma: float = EnvConsts.GAMMA) -> PPO:
    if mlp_arch is None:
        mlp_arch = [1]

    policy_kwargs = {"net_arch": [{"pi": mlp_arch, "vf": mlp_arch}], "log_std_init": log_std_init}

    logger.info("MLP architecture is: {}".format(policy_kwargs["net_arch"]))
    logger.info("gamma is: {}".format(gamma))

    ppo_model = PPO(MlpPolicy, model_envs, verbose=1, gamma=gamma, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                    policy_kwargs=policy_kwargs)

    return ppo_model

def get_json_file_from_cfg(config_folder:str):
    config_path = config_folder + "config.json"
    json_file = open(config_path, 'r')
    config = json.load(json_file)["learning"]
    json_file.close()
    return config


def load_network_and_update_env(network_file: str, env):
    net: NetworkClass = NetworkClass.load_network_object(network_file)
    env.set_network_smart_nodes_and_spr(net.get_smart_nodes, net.get_smart_nodes_spr)
    return net, env


def run_testing(model, env, num_test_observations, link_weights=None):
    env.testing(True)
    rewards_list = list()
    predict_link_weights = link_weights is None
    for _ in range(num_test_observations):
        if predict_link_weights:
            link_weights, _ = model.predict(env.reset(), deterministic=True)
        _, reward, dones, info = env.step(link_weights)
        rewards_list.append(reward * -1)

    mean_reward = np.mean(np.array(rewards_list))
    mean_reward = np.round(mean_reward, 7)
    print("Agent average performance: {}".format(mean_reward))
    return mean_reward


def greedy_best_smart_nodes_and_spr(net, traffic_matrix_list, destination_based_sprs, number_smart_nodes, smart_nodes_set):
    if smart_nodes_set is None:
        smart_nodes_set = list(filter(lambda n: len(net.out_edges_by_node(n)) > 1, net.nodes))

    smart_nodes_set = find_nodes_subsets(smart_nodes_set, number_smart_nodes)
    traffic_matrix_list = _create_random_TMs_list(traffic_matrix_list)
    matrices_mcf_LP_with_smart_nodes_solver_wrapper = partial(matrices_mcf_LP_with_smart_nodes_solver, net=net,
                                                              traffic_matrix_list=traffic_matrix_list,
                                                              destination_based_spr=destination_based_sprs)
    evaluations = list()
    headers = ["Smart Nodes Set",
               "Expected Objective"]
    data = list()
    for current_sn_set in smart_nodes_set:
        evaluations.append(matrices_mcf_LP_with_smart_nodes_solver_wrapper(current_sn_set))
        data.append(evaluations[-1][0:2])

    logger.info(tabulate(data, headers=headers))
    best_smart_nodes, best_expected_objective, best_splitting_ratios_per_src_dst_edge = min(evaluations, key=lambda t: t[1])
    logger.info("Best smart node set {} with expected objective of {}".format(best_smart_nodes, best_expected_objective))
    return best_smart_nodes, best_expected_objective, best_splitting_ratios_per_src_dst_edge


def model_learn(config_folder: str, learning_title: str, model_path: str = None, net_path: str = None, policy_updates: int = None) -> (
        PPO, RL_Smart_Nodes):
    config = get_json_file_from_cfg(config_folder)
    train_file = config_folder + config["train_file"]
    test_file = config_folder + config["test_file"]
    num_train_observations = config["num_train_observations"]
    num_test_observations = config["num_test_observations"]
    weights_factor = config["weights_factor"]
    action_weight_lb = config["weight_lb"]
    action_weight_ub = config["weight_ub"]
    n_envs = config["n_envs"]
    log_std_init = config.get("log_std_init",None)

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    n_steps = config["n_steps"]

    if policy_updates is None:
        policy_updates = config["policy_updates"]

    _envs = build_clean_smart_nodes_env(train_file, test_file, num_train_observations, num_test_observations,
                                        weights_factor=weights_factor, action_weight_lb=action_weight_lb,
                                        action_weight_ub=action_weight_ub, n_envs=n_envs)
    single_env = _envs.envs[0].env

    env_train_observations = single_env.get_train_observations

    if net_path is not None:
        load_network_and_update_env(network_file=net_path, env=single_env)

    network: NetworkClass = _envs.envs[0].env.get_network

    if model_path is not None:
        ppo_model = PPO.load(model_path, _envs)
        logger.info("********* Agent is Loaded *********")
    else:
        ppo_model = build_clean_smart_nodes_model(_envs, learning_rate, n_steps, batch_size,log_std_init=log_std_init)
        model_parameters = ppo_model.get_parameters()
        model_parameters['policy']['action_net.weight'] *= 0
        model_parameters['policy']['action_net.bias'] = torch.tensor(single_env.get_initial_weights, device=DEVICE, dtype=torch.float64)
        ppo_model.set_parameters(model_parameters)
        logger.info("********* Empty Agent is Created *********")

    total_timesteps = policy_updates * n_steps
    callback_path = CALLBACK_PATH(network.get_title) + learning_title + SEPERATOR
    checkpoint_callback = CheckpointCallback(save_freq=n_steps * 50, save_path=callback_path, name_prefix=EnvsStrings.RL_ENV_SMART_NODES_GYM_ID)
    ppo_model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    single_env.get_network.store_network_object(callback_path, env_train_observations)

    return ppo_model, single_env, weights_factor


def model_continue_learning(model: PPO, single_env: RL_Smart_Nodes, learning_title: str, policy_updates: int = None):
    n_steps = model.n_steps
    network = single_env.get_network
    total_timesteps = policy_updates * n_steps
    callback_path = CALLBACK_PATH(network.get_title) + learning_title + SEPERATOR
    checkpoint_callback = CheckpointCallback(save_freq=n_steps * 100, save_path=callback_path, name_prefix=EnvsStrings.RL_ENV_SMART_NODES_GYM_ID)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    single_env.get_network.store_network_object(callback_path, single_env.get_train_observations)
    return model, single_env
