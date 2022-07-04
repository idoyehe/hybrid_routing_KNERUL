from common.logger import logger
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gym import envs, register
from argparse import ArgumentParser
from sys import argv
import torch
import numpy as np
from platform import system

if system() == "Linux":
    assert torch.cuda.is_available()

RL_ENV_HISTORY_GYM_ID: str = 'rl-env-history-v0'


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path of the dumped file")
    parser.add_argument("-p_test", "--dumped_path_test", type=str, help="The path of the dumped test file")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value", default=0)
    parser.add_argument("-n_envs", "--number_of_envs", type=int, help="Number of vectorized environments", default=1)
    parser.add_argument("-n_steps", "--number_of_steps", type=int, help="Number of steps per ppo agent", default=384)
    parser.add_argument("-tts", "--total_timesteps", type=eval, help="Agent Total timesteps", default='384*500')
    parser.add_argument("-ep_len", "--episode_length", type=int, help="Episode Length", default=1)
    parser.add_argument("-h_len", "--history_length", type=int, help="History Length", default=0)
    parser.add_argument("-n_obs", "--number_of_observations", type=int, help="Number of observations to load",
                        default=1024)
    parser.add_argument("-s_diag", "--save_diagnostics", type=eval, help="Dump env diagnostics", default=False)
    parser.add_argument("-s_weights", "--save_links_weights", type=eval, help="Dump links weights", default=False)
    parser.add_argument("-s_agent", "--save_model_agent", type=eval, help="save the model agent", default=False)
    parser.add_argument("-l_agent", "--load_agent", type=str, help="Load a dumped agent", default=None)

    options = parser.parse_args(args)
    options.mlp_architecture = [int(layer_width) for layer_width in options.mlp_architecture.split(",")]
    return options


if __name__ == "__main__":
    args = _getOptions()
    mlp_arch = args.mlp_architecture
    gamma = args.gamma
    dumped_path = args.dumped_path
    dumped_path_test = args.dumped_path_test
    n_envs = args.number_of_envs
    n_steps = args.number_of_steps
    total_timesteps = args.total_timesteps
    episode_length = args.episode_length
    history_length = args.history_length
    num_train_observations = args.number_of_observations
    save_diagnostics = args.save_diagnostics
    save_links_weights = args.save_links_weights
    save_model_agent = args.save_model_agent
    load_agent = args.load_agent

    num_test_observations = min(num_train_observations * 2, 20000)

    checkpoint_callback = CheckpointCallback(save_freq=1000,
                                             save_path='C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\logs\\',
                                             name_prefix='rl_agent')

    logger.info("Data loaded from: {}".format(dumped_path))
    logger.info("Architecture is: {}".format(mlp_arch))
    logger.info("gamma is: {}".format(gamma))

    if RL_ENV_HISTORY_GYM_ID not in envs.registry.env_specs:
        register(id=RL_ENV_HISTORY_GYM_ID,
                 entry_point='rl_env_history:RL_Env_History',
                 kwargs={
                     'max_steps': episode_length,
                     'history_length': history_length,
                     'path_dumped': dumped_path,
                     'test_file': dumped_path_test,
                     'num_train_observations': num_train_observations,
                     'num_test_observations': num_test_observations}
                 )
    env = make_vec_env(RL_ENV_HISTORY_GYM_ID, n_envs=n_envs)
    if load_agent is None:
        policy_kwargs = [{"pi": mlp_arch, "vf": mlp_arch}]


        class CustomMLPPolicy(MlpPolicy):
            def __init__(self, *args, **kwargs):
                global policy_kwargs
                super(CustomMLPPolicy, self).__init__(*args, **kwargs, net_arch=policy_kwargs)


        model = PPO(CustomMLPPolicy, env, verbose=1, gamma=gamma, n_steps=n_steps)

        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

        env_diagnostics = env.envs[0].env.diagnostics
        if save_diagnostics:
            diag_dump_file_name = "{}_agent_diagnostics_{}".format(args.dumped_path, num_train_observations)
            diag_dump_file = open(diag_dump_file_name, 'wb')
            np.save(diag_dump_file, env_diagnostics)
            diag_dump_file.close()

    if save_model_agent and load_agent is None:
        save_path = "{}_model_agent_{}".format(dumped_path, num_train_observations)
        model.save(path=save_path)

    if load_agent is not None:
        model = PPO.load(load_agent, env)

    logger.info("Testing Part")
    env.envs[0].env.testing(True)
    obs = env.reset()
    rewards_list = list()
    diagnostics = list()
    for _ in range(num_test_observations):
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        diagnostics.extend(info)
        env.reset()
        rewards_list.append(reward[0] * -1)

    if save_links_weights:
        link_weights_file_name = "{}_agent_link_weights_{}.npy".format(args.dumped_path, num_train_observations)
        link_weights_file = open(link_weights_file_name, 'wb')
        link_weights_matrix = np.array([step_data["links_weights"] for step_data in diagnostics]).transpose()
        np.save(link_weights_file, link_weights_matrix)
        link_weights_file.close()

    rewards_file_name = "{}_agent_rewards_{}.npy".format(args.dumped_path, num_test_observations)
    rewards_file = open(rewards_file_name, 'wb')
    rewards_list = np.array(rewards_list)
    print("Average Reward: {}".format(np.mean(rewards_list)))
    np.save(rewards_file, rewards_list)
    rewards_file.close()
