import torch
from gym import envs, register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from common.logger import logger
from common.network_class import NetworkClass
from common.RL_Envs.rl_env_consts import *
from common.utils import find_nodes_subsets, SEPERATOR
from Smart_Nodes_Routing.rl_env.RL_smart_nodes import RL_Smart_Nodes
from Smart_Nodes_Routing.rl_env.smart_nodes_multiple_matrices_MCF import matrices_mcf_LP_with_smart_nodes_solver
import numpy as np
import json
from functools import partial
from tabulate import tabulate


def _create_random_TMs_list(traffic_matrices_list):
    return np.array([t[0] for t in traffic_matrices_list])


def build_clean_smart_nodes_env(train_file: str,
                                test_file: str,
                                num_train_observations: int,
                                num_test_observations: int,
                                episode_length: int = 1,
                                history_length: int = 0,
                                softMin_gamma=EnvConsts.SOFTMIN_GAMMA,
                                action_weight_lb=EnvConsts.WEIGHT_LB,
                                action_weight_ub=EnvConsts.WEIGHT_UB,
                                n_envs=2):
    logger.info("Train data loaded from: {}".format(train_file))
    logger.info("Test data loaded from: {}".format(test_file))

    logger.info("Train observations: {}".format(num_train_observations))
    logger.info("Test observations: {}".format(num_test_observations))

    if EnvsStrings.RL_ENV_SMART_NODES_GYM_ID in envs.registry.env_specs:
        del envs.registry.env_specs[EnvsStrings.RL_ENV_SMART_NODES_GYM_ID]

    register(id=EnvsStrings.RL_ENV_SMART_NODES_GYM_ID,
             entry_point='Smart_Nodes_Routing.rl_env.RL_smart_nodes:RL_Smart_Nodes',
             kwargs={
                 'max_steps': episode_length,
                 'history_length': history_length,
                 'path_dumped': train_file,
                 'test_file': test_file,
                 'num_train_observations': num_train_observations,
                 'num_test_observations': num_test_observations,
                 'softMin_gamma': softMin_gamma,
                 'action_weight_lb': action_weight_lb,
                 'action_weight_ub': action_weight_ub})

    return make_vec_env(EnvsStrings.RL_ENV_SMART_NODES_GYM_ID, n_envs=n_envs)


def build_clean_smart_nodes_model(model_envs, learning_rate: float, n_steps: int,
                                  batch_size: int,
                                  mlp_arch=None,
                                  gamma: float = EnvConsts.GAMMA) -> PPO:
    if mlp_arch is None:
        mlp_arch = [1]

    policy_kwargs = {"net_arch": [{"pi": mlp_arch, "vf": mlp_arch}]}

    logger.info("MLP architecture is: {}".format(policy_kwargs["net_arch"]))
    logger.info("gamma is: {}".format(gamma))

    ppo_model = PPO(MlpPolicy, model_envs, verbose=1, gamma=gamma, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                    policy_kwargs=policy_kwargs)
    par = ppo_model.get_parameters()
    par['policy']['log_std'] += -10
    par['policy']['action_net.weight'] = torch.zeros_like(par['policy']['action_net.weight'])

    par['policy']['action_net.bias'] = torch.tensor(
        [21.96883, 11.56587, 11.01115, 8.46576, 12.81978, 6.82846, 9.67489, 5.27979, 5.25515, 5.28384, 5.27219, 5.26027, 6.74640, 22.91519, 13.97877,
         22.88946, 14.47366, 10.37872, 5.16104, 7.79711, 5.16648, 5.14826, 10.83628, 19.70354, 9.96238, 8.76365, 5.14975, 5.18224, 6.64693, 14.79913,
         5.15943, 5.17707, 19.81461, 11.51679, 8.92476, 5.09122, 9.75154, 6.37023, 7.79810, 10.89047, 5.11598, 5.08135, 5.08888, 10.30671, 9.62940,
         5.09141, 10.06301, 13.94150, 8.68659, 12.91445, 10.28453, 6.86186, 6.44171, 7.68533, 9.59151, 7.82101, 6.67048, 5.15015, 5.18239, 5.16146,
         8.31751, 9.21555, 10.09108, 6.72051, 15.74292, 10.08667, 17.93862, 5.11609, 7.47548, 8.37816, 6.64782, 5.16642, 6.93324, 6.15637, 5.28003,
         5.25521, 5.28422, 5.27248, 5.15973, 5.08166, 5.26045, 5.17725, 6.88432, 17.80110, 5.14854, 5.08880
         ], dtype=torch.float32, device='cpu')
    # par['policy']['action_net.bias'] = torch.tensor([8.748] * 86, dtype=torch.float32, device='cpu')
    ppo_model.set_parameters(par)

    return ppo_model


def load_network_and_update_env(network_file: str, env):
    net: NetworkClass = NetworkClass.load_network_object(network_file)
    env.set_network_smart_nodes_and_spr(net.get_smart_nodes, net.get_smart_nodes_spr)
    env.set_train_observations(net.env_train_observation)

    return net, env


def run_testing(model, env, num_test_observations):
    env.testing(True)
    rewards_list = list()
    for _ in range(num_test_observations):
        obs = env.reset()
        link_weights, _ = model.predict(env.reset(), deterministic=True)
        _, reward, dones, info = env.step(link_weights)
        rewards_list.append(reward)

    mean_reward = np.mean(rewards_list)
    print("Agent average performance: {}".format(mean_reward * -1))
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
    config_path = config_folder + "config.json"
    json_file = open(config_path, 'r')
    config = json.load(json_file)["learning"]
    json_file.close()
    train_file = config_folder + config["train_file"]
    test_file = config_folder + config["test_file"]
    num_train_observations = config["num_train_observations"]
    num_test_observations = config["num_test_observations"]
    softMin_gamma = config["softMin_gamma"]
    action_weight_lb = config["weight_lb"]
    action_weight_ub = config["weight_ub"]

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    n_steps = config["n_steps"]

    if policy_updates is None:
        policy_updates = config["policy_updates"]

    _envs = build_clean_smart_nodes_env(train_file, test_file, num_train_observations, num_test_observations,
                                        softMin_gamma=softMin_gamma, action_weight_lb=action_weight_lb, action_weight_ub=action_weight_ub)
    single_env = _envs.envs[0].env

    env_train_observations = single_env.get_train_observations

    if net_path is not None:
        load_network_and_update_env(network_file=net_path, env=single_env)

    network: NetworkClass = _envs.envs[0].env.get_network

    if model_path is not None:
        model = PPO.load(model_path, _envs)
        logger.info("********* Agent is Loaded *********")
    else:
        model = build_clean_smart_nodes_model(_envs, learning_rate, n_steps, batch_size)
        logger.info("********* Empty Agent is Created *********")

    total_timesteps = policy_updates * n_steps
    callback_path = CALLBACK_PATH(network.get_title) + learning_title + SEPERATOR
    checkpoint_callback = CheckpointCallback(save_freq=n_steps * 100, save_path=callback_path, name_prefix=EnvsStrings.RL_ENV_SMART_NODES_GYM_ID)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    single_env.get_network.store_network_object(callback_path, env_train_observations)

    return model, single_env


def model_continue_learning(model: PPO, single_env: RL_Smart_Nodes, learning_title: str, policy_updates: int = None):
    n_steps = model.n_steps
    network = single_env.get_network
    total_timesteps = policy_updates * n_steps
    callback_path = CALLBACK_PATH(network.get_title) + learning_title + SEPERATOR
    checkpoint_callback = CheckpointCallback(save_freq=n_steps * 100, save_path=callback_path, name_prefix=EnvsStrings.RL_ENV_SMART_NODES_GYM_ID)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    single_env.get_network.store_network_object(callback_path, single_env.get_train_observations)
    return model, single_env
