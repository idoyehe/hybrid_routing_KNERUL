from gym import envs, register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from common.logger import logger
from common.topologies import topology_zoo_loader
from common.network_class import NetworkClass
from common.RL_Envs.rl_env_consts import *
from common.utils import find_nodes_subsets, SEPERATOR, load_dump_file
from Smart_Nodes_Routing.rl_env.RL_smart_nodes import RL_Smart_Nodes
from Smart_Nodes_Routing.rl_env.smart_nodes_multiple_matrices_MCF import matrices_mcf_LP_with_smart_nodes_solver
from Link_State_Routing_PEFT.gradiant_decent.original_PEFT import PEFT_main_loop
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
                                weights_factor=EnvConsts.WEIGHTS_FACTOR,
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
                 'weights_factor': weights_factor,
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
    weights_factor = config["weights_factor"]
    action_weight_lb = config["weight_lb"]
    action_weight_ub = config["weight_ub"]

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    n_steps = config["n_steps"]

    if policy_updates is None:
        policy_updates = config["policy_updates"]

    _envs = build_clean_smart_nodes_env(train_file, test_file, num_train_observations, num_test_observations,
                                        weights_factor=weights_factor, action_weight_lb=action_weight_lb, action_weight_ub=action_weight_ub)
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


def get_initial_weights(config_folder: str):
    config_path = config_folder + "config.json"
    json_file = open(config_path, 'r')
    config = json.load(json_file)["learning"]
    json_file.close()
    train_file = config_folder + config["train_file"]
    loaded_dict = load_dump_file(train_file)
    traffic_matrix = np.sum(tm for tm, _, _ in loaded_dict["tms"])
    necessary_capacity = loaded_dict["necessary_capacity_per_tm"][-1]
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))
    w_u_v, PEFT_congestion = PEFT_main_loop(net, traffic_matrix, necessary_capacity, 1)
    return w_u_v


if __name__ == "__main__":
    from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
    from Learning_to_Route.rl_softmin_history.soft_min_optimizer import SoftMinOptimizer
    from common.static_routing.multiple_matrices_MCF import multiple_tms_mcf_LP_solver

    config_folder = "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\common\\TMs_DB\\ScaleFree70Nodes\\"
    # w_u_v = get_initial_weights(config_folder)
    w_u_v = np.array(
        [3.28489, 3.84911, 2.40156, 2.53162, 2.08734, 2.81310, 2.71351, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 2.27188, 3.34352, 3.31877,
         3.52526, 2.16697, 1.91787, 10.00000, 2.50944, 10.00000, 10.00000, 3.79226, 3.29915, 3.53814, 2.81792, 10.00000, 10.00000, 2.09823, 2.43083,
         10.00000, 10.00000, 3.56389, 3.48897, 2.20534, 10.00000, 2.65600, 2.33144, 2.74463, 2.57577, 10.00000, 10.00000, 10.00000, 2.39145, 2.21545,
         10.00000, 2.52263, 2.22938, 2.60257, 2.08059, 1.92462, 2.31761, 1.09414, 2.80892, 2.84131, 2.72573, 1.81322, 10.00000, 10.00000, 10.00000,
         2.11520, 2.64129, 1.72426, 2.47989, 2.36318, 1.80674, 1.68575, 10.00000, 2.72704, 1.81353, 1.47456, 10.00000, 1.08030, 1.48839, 10.00000,
         10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 2.28660, 1.67103, 10.00000, 10.00000
         ], dtype=np.float64)

    config_path = config_folder + "config.json"
    json_file = open(config_path, 'r')
    config = json.load(json_file)["learning"]
    json_file.close()
    train_file = config_folder + config["train_file"]
    loaded_dict = load_dump_file(train_file)
    traffic_matrix_list = loaded_dict["tms"]
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))
    traffic_distribution = PEFTOptimizer(net, -1)

    dst_splitting_ratios = traffic_distribution.calculating_destination_based_spr(w_u_v)
    a = [traffic_distribution.step(w_u_v, tm, opt)[0] for tm, opt, _ in traffic_matrix_list]
    print(np.mean(a))
    best_smart_nodes, best_expected_objective, best_splitting_ratios_per_src_dst_edge = greedy_best_smart_nodes_and_spr(net,
                                                                                                                        traffic_matrix_list,
                                                                                                                        dst_splitting_ratios, 3, (0,1,2,3))
