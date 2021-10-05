from gym import envs, register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from common.logger import logger
from common.network_class import NetworkClass
from common.RL_Envs.rl_env_consts import *
from common.utils import find_nodes_subsets
from Smart_Nodes_Routing.rl_env.RL_smart_nodes import RL_Smart_Nodes
from smart_nodes_multiple_matrices_MCF import matrices_mcf_LP_with_smart_nodes_solver
import numpy as np
from multiprocessing import Pool
from functools import partial
from tabulate import tabulate


def build_clean_smart_nodes_env(train_file: str,
                                test_file: str,
                                num_train_observations: int,
                                num_test_observations: int,
                                episode_length: int = 1,
                                history_length: int = 0,
                                softMin_gamma=EnvConsts.SOFTMIN_GAMMA,
                                action_weight_lb=EnvConsts.WEIGHT_LB,
                                action_weight_ub=EnvConsts.WEIGHT_UB,
                                action_weight_factor=EnvConsts.WEIGHT_FACTOR,
                                n_envs=1):
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
                 'action_weight_ub': action_weight_ub,
                 'action_weight_factor': action_weight_factor})

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
    smart_nodes_set.append(tuple())
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
