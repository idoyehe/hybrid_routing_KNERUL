# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import ecmp_history
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from argparse import ArgumentParser
from sys import argv


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-p", "--save_path", type=str, help="The path to save the model")
    parser.add_argument("-arch", "--mlp_architecture", type=str, help="The architecture of the neural network")
    parser.add_argument("-gamma", "--gamma", type=float, help="Gamma Value")
    options = parser.parse_args(args)
    options.mlp_architecture = [int(layer_width) for layer_width in options.mlp_architecture.split(",")]
    return options


#
# # for i in range(0, 6):
# # with model.graph.as_default():
# #     saver = tf.compat.v1.train.Saver()
# #     saver.save(training_sess, "./pcc_model_%d.ckpt" % i)
# model.learn(total_timesteps=(100 * 8192))
#
# env.testing(True)
# obs = env.reset()
# for _ in range(10 * 8192):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#
# env.reset()


class MyMlpPolicy(MlpPolicy):
    ARCH = None

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          net_arch=[{"pi": MyMlpPolicy.ARCH, "vf": MyMlpPolicy.ARCH}],
                                          **_kwargs)


if __name__ == "__main__":
    args = _getOptions()
    MyMlpPolicy.ARCH = args.mlp_architecture

    print("Architecture is: {}".format(MyMlpPolicy.ARCH))
    gamma = args.gamma
    print("gamma = {}".format(gamma))

    save_path = args.save_path

    env = make_vec_env(ecmp_history.ECMP_ENV_GYM_ID, n_envs=1)
    model = PPO(MlpPolicy, env, verbose=1, gamma=gamma, n_steps=350)

    # model.learn(total_timesteps=(7 * 50 * 1500))
    model.learn(total_timesteps=(20))
    # model.save(path=save_path)
