import ecmp_history_simplified
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from gym import envs, register
from common.rl_env_consts import HistoryConsts
from argparse import ArgumentParser
from sys import argv
import torch

assert torch.cuda.is_available()

ECMP_ENV_GYM_ID: str = 'ecmp-history-simplified-v0'


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path of the dumped file")
    parser.add_argument("-topo", "--topo_gml_path", type=str, help="The path of the topology GML file")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value", default=0)
    parser.add_argument("-n_envs", "--number_of_envs", type=int, help="Number of vectorized environments", default=1)
    parser.add_argument("-n_steps", "--number_of_steps", type=int, help="Number of steps per ppo agent", default=100)
    parser.add_argument("-tts", "--total_timesteps", type=str, help="Agent Total timesteps", default="1000")
    parser.add_argument("-ep_len", "--episode_length", type=int, help="Episode Length", default=1)
    parser.add_argument("-h_len", "--history_length", type=int, help="History Length", default=10)
    parser.add_argument("-n_matrices", "--number_of_matrices", type=int, help="Number of matrices to load", default=350)
    parser.add_argument("-bt_th", "--bottleneck_threshold", type=float, help="STD Threshold", default=0.4)

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
    topo = args.topo_gml_path
    n_envs = args.number_of_envs
    n_steps = args.number_of_steps
    total_timesteps = args.total_timesteps
    episode_length = args.episode_length
    history_length = args.history_length
    number_of_matrices = args.number_of_matrices
    bottleneck_threshold = args.bottleneck_threshold

    save_path = "{}_agent_simplified_{}".format(args.dumped_path, number_of_matrices)
    dump_file_name = "{}_agent_diagnostics_{}".format(args.dumped_path, number_of_matrices)

    if ECMP_ENV_GYM_ID not in envs.registry.env_specs:
        register(id=ECMP_ENV_GYM_ID,
                 entry_point='ecmp_history_simplified:ECMPHistorySimplifiedEnv',
                 kwargs={
                     'max_steps': episode_length,
                     'bottleneck_th':bottleneck_threshold,
                     'history_length': history_length,
                     'topo_customize': topo,
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

    model.save(path=save_path)
    env.close()
