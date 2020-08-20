import ecmp_history
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from argparse import ArgumentParser
from sys import argv
import pickle
import torch

assert torch.cuda.is_available()


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--save_path", type=str, help="The path to save the model")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value", default=0)
    parser.add_argument("-n_envs", "--number_of_envs", type=int, help="Number of vectorized environments", default=1)
    options = parser.parse_args(args)
    options.mlp_architecture = [int(layer_width) for layer_width in options.mlp_architecture.split(",")]
    return options


if __name__ == "__main__":
    args = _getOptions()

    print("Architecture is: {}".format(args.mlp_architecture))
    gamma = args.gamma
    print("gamma = {}".format(gamma))

    save_path = "{}_agent".format(args.save_path)
    dump_file_name = "{}_agent_diagnostics".format(args.save_path)
    n_envs = args.number_of_envs

    env = make_vec_env(ecmp_history.ECMP_ENV_GYM_ID, n_envs=n_envs)
    policy_kwargs = dict(net_arch=args.mlp_architecture)

    model = PPO(MlpPolicy, env, verbose=1, gamma=gamma, n_steps=50 * 7, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=(50 * 7 * 1500 * 500))
    all_envs_diagnostics = []
    for env_data in env.envs:
        all_envs_diagnostics.append(env_data.env.diagnostics)
    model.save(path=save_path)
    env.close()

    dump_file = open(dump_file_name, 'wb')
    pickle.dump({"agent_diagnostics": all_envs_diagnostics}, dump_file)
    dump_file.close()
