from common.utils import load_dump_file
from common.network_class import NetworkClass
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym import envs, register
from stable_baselines3.common.env_util import make_vec_env
from argparse import ArgumentParser
from sys import argv
from platform import system
from experiments.RL_smart_nodes import RL_Smart_Nodes
from experiments.smart_nodes_multiple_matrices_MCF import *
from multiprocessing import Pool
import torch
import numpy as np

IS_LINUX = system() == "Linux"

if IS_LINUX:
    assert torch.cuda.is_available()

RL_ENV_SMART_NODES_GYM_ID: str = 'rl-smart_nodes-v0'


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path of the dumped file")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network", default="1")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value", default=0)
    parser.add_argument("-n_steps", "--number_of_steps", type=int, help="Number of steps per ppo agent", default=100)
    parser.add_argument("-tts", "--total_timesteps", type=str, help="Agent Total timesteps", default="1000")
    parser.add_argument("-ep_len", "--episode_length", type=int, help="Episode Length", default=1)
    parser.add_argument("-h_len", "--history_length", type=int, help="History Length", default=0)
    parser.add_argument("-n_obs", "--number_of_observations", type=int, help="Number of observations to load", default=350)
    parser.add_argument("-s_weights", "--save_links_weights", type=eval, help="Dump links weights", default=False)
    parser.add_argument("-s_agent", "--save_model_agent", type=eval, help="save the model agent", default=False)
    parser.add_argument("-l_agent", "--load_agent", type=str, help="Load a dumped agent", default=None)
    parser.add_argument("-n_iter", "--number_of_iterations", type=int, help="Number of iteration", default=2)
    parser.add_argument("-sample_size", "--tms_sample_size", type=int, help="Percent of smart nodes", default=50)

    options = parser.parse_args(args)
    options.total_timesteps = eval(options.total_timesteps)
    options.mlp_architecture = [int(layer_width) for layer_width in options.mlp_architecture.split(",")]
    return options


def return_best_smart_nodes_and_spr(net, traffic_matrix_list, dest_spr, current_smart_nodes):
    smart_nodes_set = []
    for node in net.nodes:
        if node in current_smart_nodes:
            continue
        else:
            smart_nodes_set.append(current_smart_nodes + (node,))
    params = [(net, traffic_matrix_list, dest_spr, smart_nodes) for smart_nodes in smart_nodes_set]

    pool = Pool(processes=len(smart_nodes_set))
    results = pool.starmap(func=matrices_mcf_LP_with_smart_nodes_solver, iterable=params)
    return min(results, key=lambda t: t[0])


if __name__ == "__main__":
    args = _getOptions()
    mlp_arch = args.mlp_architecture
    gamma = args.gamma
    dumped_path = args.dumped_path
    n_steps = args.number_of_steps
    total_timesteps = args.total_timesteps
    episode_length = args.episode_length
    history_length = args.history_length
    num_train_observations = args.number_of_observations
    save_links_weights = args.save_links_weights
    save_model_agent = args.save_model_agent
    load_agent = args.load_agent
    num_of_iterations = args.number_of_iterations
    tms_sample_size = args.tms_sample_size

    num_test_observations = min(num_train_observations * 2, 20000)

    callback_perfix_path = '/home/idoye/PycharmProjects/Research_Implementing/experiments/callbacks/' \
        if IS_LINUX else "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\experiments\\callbacks\\"

    logger.info("Data loaded from: {}".format(dumped_path))
    logger.info("Architecture is: {}".format(mlp_arch))
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
    envs = make_vec_env(RL_ENV_SMART_NODES_GYM_ID)
    env = envs.envs[0].env

    loaded_dict = load_dump_file(dumped_path)
    net = env.get_network

    if load_agent is not None:
        model = PPO.load(load_agent, envs)
        logger.info("Iteration 0 Starts, model is loaded...")


    else:
        assert load_agent is None
        policy_kwargs = [{"pi": mlp_arch, "vf": mlp_arch}]


        class CustomMLPPolicy(MlpPolicy):
            def __init__(self, *args, **kwargs):
                global policy_kwargs
                super(CustomMLPPolicy, self).__init__(*args, **kwargs, net_arch=policy_kwargs)


        model = PPO(CustomMLPPolicy, envs, verbose=1, gamma=gamma, n_steps=n_steps)

        logger.info("Iteration 0 Starts, model is learning...")
        env.testing(False)
        callback_path = callback_perfix_path + "iteration_{}".format(0) + ("/" if IS_LINUX else "\\")
        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps, save_path=callback_path, name_prefix=RL_ENV_SMART_NODES_GYM_ID)
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    current_smart_nodes = tuple()
    for i in range(num_of_iterations):
        logger.info("Iteration {}, model is predicting...".format(i))
        env.testing(True)
        link_weights, _ = model.predict(env.reset(), deterministic=True)
        traffic_matrix_list = create_weighted_traffic_matrices(tms_sample_size, loaded_dict["tms"])  # create a samples from the tms distribution
        dest_spr = env.get_optimizer.calculating_splitting_ratios(link_weights)

        logger.info("Iteration {}, evaluating smart nodes...".format(i))
        best_smart_nodes = return_best_smart_nodes_and_spr(net, traffic_matrix_list, dest_spr, current_smart_nodes)
        logger.info("Iteration {}, Chosen smart nodes: {}".format(i, best_smart_nodes[1]))
        current_smart_nodes = best_smart_nodes[1]
        env.set_network_smart_nodes_and_spr(current_smart_nodes, best_smart_nodes[2])

        env.get_network.store_network_object(callback_path)


        total_timesteps /= 2
        logger.info("Iteration {}, model is learning...".format(i + 1))
        env.testing(False)

        callback_path = callback_perfix_path + "iteration_{}".format(i) + ("/" if IS_LINUX else "\\")
        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps, save_path=callback_path, name_prefix=RL_ENV_SMART_NODES_GYM_ID)
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    logger.info("Iterations Done!!")

    obs = env.reset()
    rewards_list = list()
    diagnostics = list()
    for _ in range(num_test_observations):
        link_weights, _ = model.predict(env.reset(), deterministic=True)
        _, reward, dones, info = env.step(link_weights)
        diagnostics.extend(info)
        obs = env.reset()
        rewards_list.append(reward * -1)

    if save_links_weights:
        link_weights_file_name = "{}_links_weights_{}.npy".format(args.dumped_path, num_train_observations)
        link_weights_file = open(link_weights_file_name, 'wb')
        link_weights_matrix = np.array([step_data["links_weights"] for step_data in diagnostics]).transpose()
        np.save(link_weights_file, link_weights_matrix)
        link_weights_file.close()

    rewards_file_name = "{}_agent_rewards_{}.npy".format(args.dumped_path, num_test_observations)
    rewards_file = open(rewards_file_name, 'wb')
    rewards_list = np.array(rewards_list)
    np.save(rewards_file, rewards_list)
    rewards_file.close()
