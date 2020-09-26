import ecmp_history
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from gym import envs, register
from common.rl_env_consts import HistoryConsts
from argparse import ArgumentParser
from sys import argv
import pickle
import torch
import numpy as np

assert torch.cuda.is_available()

ECMP_ENV_GYM_ID: str = 'ecmp-history-v0'


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path of the dumped file")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value", default=0)
    parser.add_argument("-n_envs", "--number_of_envs", type=int, help="Number of vectorized environments", default=1)
    parser.add_argument("-n_steps", "--number_of_steps", type=int, help="Number of steps per ppo agent", default=100)
    parser.add_argument("-tts", "--total_timesteps", type=str, help="Agent Total timesteps", default="1000")
    parser.add_argument("-ep_len", "--episode_length", type=int, help="Episode Length", default=1)
    parser.add_argument("-h_len", "--history_length", type=int, help="History Length", default=10)
    parser.add_argument("-n_matrices", "--number_of_matrices", type=int, help="Number of matrices to load",
                        default=350)
    parser.add_argument("-s_diag", "--save_diagnostics", type=bool, help="Dump env diagnostics", default=False)
    parser.add_argument("-s_weights", "--save_links_weights", type=bool, help="Dump links weights", default=False)

    options = parser.parse_args(args)
    options.total_timesteps = eval(options.total_timesteps)
    options.mlp_architecture = [int(layer_width) for layer_width in options.mlp_architecture.split(",")]
    return options


if __name__ == "__main__":
    args = _getOptions()

    print("Architecture is: {}".format(args.mlp_architecture))
    gamma = args.gamma
    print("gamma = {}".format(gamma))
    dumped_path = args.dumped_path
    n_envs = args.number_of_envs
    n_steps = args.number_of_steps
    total_timesteps = args.total_timesteps
    episode_length = args.episode_length
    history_length = args.history_length
    number_of_matrices = args.number_of_matrices
    save_diagnostics = args.save_diagnostics
    save_links_weights = args.save_links_weights

    save_path = "{}_agent_{}".format(args.save_path, number_of_matrices)
    dump_file_name = "{}_agent_diagnostics_{}".format(args.save_path, number_of_matrices)

    if ECMP_ENV_GYM_ID not in envs.registry.env_specs:
        register(id=ECMP_ENV_GYM_ID,
                 entry_point='ecmp_history:ECMPHistoryEnv',
                 kwargs={
                     'max_steps': episode_length,
                     'history_length': history_length,
                     'path_dumped': dumped_path,
                     'train_histories_length': number_of_matrices,
                     'test_histories_length': 0,
                     'history_action_type': HistoryConsts.ACTION_W_EPSILON}
                 )

    env = make_vec_env(ECMP_ENV_GYM_ID, n_envs=n_envs)
    policy_kwargs = [{"pi": args.mlp_architecture, "vf": args.mlp_architecture}]


    class CustomMLPPolicy(MlpPolicy):
        def __init__(self, *args, **kwargs):
            global policy_kwargs
            super(CustomMLPPolicy, self).__init__(*args, **kwargs, net_arch=policy_kwargs)


    model = PPO(CustomMLPPolicy, env, verbose=1, gamma=gamma, n_steps=n_steps)

    model.learn(total_timesteps=total_timesteps)

    env_diagnostics = env.envs[0].env.diagnostics
    if save_diagnostics:
        dump_file = open(dump_file_name, 'wb')
        pickle.dump({"env_diagnostics": env_diagnostics}, dump_file)
        dump_file.close()

    model.save(path=save_path)
    env.close()

    if save_links_weights:
        link_weights_file_name = "{}_agent_link_weights_{}.npy".format(args.save_path, number_of_matrices)
        link_weights_file = open(link_weights_file_name, 'wb')
        link_weights_matrix = np.array([step_data["links_weights"] for step_data in env_diagnostics]).transpose()
        np.save(link_weights_file, link_weights_matrix)
        link_weights_file.close()
