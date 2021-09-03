from common.utils import load_dump_file, find_nodes_subsets
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym import envs, register
from stable_baselines3.common.env_util import make_vec_env
from argparse import ArgumentParser
from sys import argv
from platform import system
from experiments.smart_nodes_multiple_matrices_MCF import *
from multiprocessing import Pool
import torch
import numpy as np
from functools import partial
from tabulate import tabulate
from experiments.RL_smart_nodes import RL_Smart_Nodes

IS_LINUX = system() == "Linux"

if IS_LINUX:
    assert torch.cuda.is_available()

RL_ENV_SMART_NODES_GYM_ID: str = 'rl-smart_nodes-v0'


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path of the dumped file")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network",
                        default="1")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value", default=0)
    parser.add_argument("-n_steps", "--number_of_steps", type=int, help="Number of steps per ppo agent", default=100)
    parser.add_argument("-p_updates", "--policy_updates", type=int, help="number of policy update, or tts")
    parser.add_argument("-ep_len", "--episode_length", type=int, help="Episode Length", default=1)
    parser.add_argument("-h_len", "--history_length", type=int, help="History Length", default=0)
    parser.add_argument("-n_obs", "--number_of_observations", type=int, help="Number of observations to load",
                        default=350)
    parser.add_argument("-l_agent", "--load_agent", type=str, help="Load a dumped agent", default=None)
    parser.add_argument("-l_net", "--load_network", type=str, help="Load a dumped Network object", default=None)
    parser.add_argument("-n_iter", "--number_of_iterations", type=int, help="Number of iteration", default=2)
    parser.add_argument("-sample_size", "--tms_sample_size", type=int, help="Batch Size", default=200)
    parser.add_argument("-prcs", "--processes", type=int, help="Number of Processes", default=1)
    parser.add_argument("-n_sn", "--number_smart_nodes", type=int, help="Number of smart nodes", default=1)
    parser.add_argument("-s_nodes", "--smart_nodes_set", type=eval, help="Smart Node set to examine", default=None)
    parser.add_argument("-kpp", "--kpp", type=eval, help="Use KPP to choose greedy", default=False)

    options = parser.parse_args(args)
    options.mlp_architecture = [int(layer_width) for layer_width in options.mlp_architecture.split(",")]
    return options


def greedy_best_smart_nodes_and_spr(net, traffic_matrix_list, destination_based_sprs, number_smart_nodes,
                                    smart_nodes_set, processes=1):
    if smart_nodes_set is None:
        smart_nodes_set = list(filter(lambda n: len(net.out_edges_by_node(n)) > 1, net.nodes))

    smart_nodes_set = find_nodes_subsets(smart_nodes_set, number_smart_nodes)
    smart_nodes_set.append(tuple())
    matrices_mcf_LP_with_smart_nodes_solver_wrapper = partial(matrices_mcf_LP_with_smart_nodes_solver, net=net,
                                                              traffic_matrix_list=traffic_matrix_list,
                                                              destination_based_spr=destination_based_sprs)
    evaluations = list()
    start_idx = 0
    headers = ["Smart Nodes Set",
               "Expected Objective"]
    data = list()
    while start_idx < len(smart_nodes_set):
        if processes > 1:
            end_idx = min(start_idx + processes, len(smart_nodes_set))
            pool = Pool(processes=end_idx - start_idx)
            evaluations += pool.map(func=matrices_mcf_LP_with_smart_nodes_solver_wrapper,
                                    iterable=smart_nodes_set[start_idx:end_idx])
            start_idx += processes
        else:
            assert processes <= 1
            evaluations.append(matrices_mcf_LP_with_smart_nodes_solver_wrapper(smart_nodes_set[start_idx]))
            data.append(evaluations[-1][0:2])
            start_idx += 1

    logger.info(tabulate(data, headers=headers))

    return min(evaluations, key=lambda t: t[1])


if __name__ == "__main__":
    args = _getOptions()
    mlp_arch = args.mlp_architecture
    gamma = args.gamma
    dumped_path = args.dumped_path
    n_steps = args.number_of_steps
    policy_updates = args.policy_updates
    episode_length = args.episode_length
    history_length = args.history_length
    num_train_observations = args.number_of_observations
    load_agent = args.load_agent
    load_network = args.load_network
    num_of_iterations = args.number_of_iterations
    tms_sample_size = args.tms_sample_size
    number_smart_nodes = args.number_smart_nodes
    smart_nodes_set = args.smart_nodes_set
    processes = args.processes
    kpp = args.kpp

    total_timesteps = policy_updates * n_steps
    num_test_observations = min(num_train_observations * 5, 20000)

    logger.info("Data loaded from: {}".format(dumped_path))
    logger.info("Architecture is: {}".format(mlp_arch))
    logger.info("Policy updates: {}".format(policy_updates))
    logger.info("Train observations: {}".format(num_train_observations))
    logger.info("gamma is: {}".format(gamma))

    if RL_ENV_SMART_NODES_GYM_ID not in envs.registry.env_specs:
        register(id=RL_ENV_SMART_NODES_GYM_ID,
                 # entry_point='rl_env_history:RL_Env_History',
                 entry_point='RL_smart_nodes:RL_Smart_Nodes',
                 kwargs={
                     'max_steps': episode_length,
                     'history_length': history_length,
                     'path_dumped': dumped_path,
                     'num_train_observations': num_train_observations,
                     'num_test_observations': num_test_observations}
                 )
    envs = make_vec_env(RL_ENV_SMART_NODES_GYM_ID, n_envs=1)
    env = envs.envs[0].env
    env_train_observations = env.get_train_observations


    loaded_dict = load_dump_file(dumped_path)
    if load_network is None:
        net: NetworkClass = env.get_network
    else:
        net: NetworkClass = NetworkClass.load_network_object(load_network)
        env.set_network_smart_nodes_and_spr(net.get_smart_nodes, net.get_smart_nodes_spr)
        env.set_train_observations(net.env_train_observation)

    callback_perfix_path = '/home/idoye/PycharmProjects/Research_Implementing/experiments/{}_callbacks_batch/'.format(
        net.get_title) \
        if IS_LINUX else "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\experiments\\{}_callbacks_batch\\".format(
        net.get_title)

    if load_agent is not None:
        model = PPO.load(load_agent, envs)
        logger.info("********* Iteration 0 Starts, Agent is Loaded *********")


    else:
        assert load_agent is None
        policy_kwargs = [{"pi": mlp_arch, "vf": mlp_arch}]


        class CustomMLPPolicy(MlpPolicy):
            def __init__(self, *args, **kwargs):
                global policy_kwargs
                super(CustomMLPPolicy, self).__init__(*args, **kwargs, net_arch=policy_kwargs)


        model = PPO(CustomMLPPolicy, envs, verbose=1, gamma=gamma, n_steps=n_steps, batch_size=n_steps)

        logger.info("********* Iteration 0 Starts, Agent is learning *********")
        callback_path = callback_perfix_path + "iteration_{}".format(0) + ("/" if IS_LINUX else "\\")
        checkpoint_callback = CheckpointCallback(save_freq=n_steps * 100, save_path=callback_path,
                                                 name_prefix=RL_ENV_SMART_NODES_GYM_ID)
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        env.get_network.store_network_object(callback_path,env_train_observations)
        logger.info("***************** Iteration 0 Finished ******************")

    current_smart_nodes = tuple()
    for i in range(1, num_of_iterations + 1):
        logger.info("**** Iteration {}, Evaluating Smart Node  *****".format(i))
        link_weights, _ = model.predict(env.reset(), deterministic=True)

        traffic_matrix_list = create_random_TMs_list(tms_sample_size, loaded_dict["tms"], shuffling=True)
        destination_based_sprs = env.get_optimizer.calculating_destination_based_spr(link_weights)

        if kpp:
            kp_set = env.get_optimizer.key_player_problem_comm_iter(link_weights, number_smart_nodes)
            best_smart_nodes = matrices_mcf_LP_with_smart_nodes_solver(kp_set, env.get_network, traffic_matrix_list, destination_based_sprs)
            logger.info("********** Iteration {}, KPP Expected Objective:{}  ***********".format(i, best_smart_nodes[1]))


        else:
            best_smart_nodes = greedy_best_smart_nodes_and_spr(net, traffic_matrix_list, destination_based_sprs, number_smart_nodes, smart_nodes_set,
                                                               processes)
        current_smart_nodes = best_smart_nodes[0]
        env.set_network_smart_nodes_and_spr(current_smart_nodes, best_smart_nodes[2])
        logger.info("********** Iteration {}, Smart Nodes:{}  ***********".format(i, current_smart_nodes))

        logger.info("********* Iteration {} Starts, Agent is learning *********".format(i))

        total_timesteps /= 2
        callback_path = callback_perfix_path + "iteration_{}".format(i) + ("/" if IS_LINUX else "\\")
        checkpoint_callback = CheckpointCallback(save_freq=n_steps * 100, save_path=callback_path,
                                                 name_prefix=RL_ENV_SMART_NODES_GYM_ID)
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        env.get_network.store_network_object(callback_path,env_train_observations)

    logger.info("========================== Learning Process is Done =================================")

    env.testing(True)
    rewards_list = list()
    diagnostics = list()
    for _ in range(num_test_observations):
        obs = env.reset()
        link_weights, _ = model.predict(env.reset(), deterministic=True)
        _, reward, dones, info = env.step(link_weights)
        diagnostics.extend(info)
        rewards_list.append(reward * -1)

    print("Agent average performance: {}".format(np.mean(rewards_list)))
