"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 20 Dec 2020
@by: Ido Yehezkel
"""

from rl_env import *
from Learning_to_Route.common.utils import error_bound
from optimizer_oblivious import WNumpyOptimizer_Oblivious


class RL_Env_Oblivious(RL_Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):
        super(RL_Env_Oblivious, self).__init__(max_steps=max_steps, path_dumped=path_dumped,
                                               history_length=history_length,
                                               history_action_type=history_action_type,
                                               num_train_observations=num_train_observations,
                                               num_test_observations=num_test_observations, testing=testing)

        self._num_edges = self._network.get_num_edges
        self._optimizer = WNumpyOptimizer_Oblivious(self._network)
        self._set_action_space()

        self._diagnostics = list()

    @property
    def diagnostics(self):
        return np.array(self._diagnostics)

    def _set_action_space(self):
        self._action_space = spaces.Box(low=0, high=np.inf, shape=(self._num_edges,))

    def step(self, action):
        info = dict()
        links_weights = self._modify_action(action)

        kl_value = self._process_action_get_cost(links_weights)
        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        if self._testing:
            info["links_weights"] = np.array(links_weights)

        del links_weights
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()

        reward = kl_value * self._NORM_FACTOR
        done = self._is_terminal

        return observation, reward, done, info

    def reset(self):
        self._tm_start_index = 0
        self._current_observation_index = (self._current_observation_index + 1) % self._observations_length
        return self._get_observation()

    def _process_action_get_cost(self, links_weights):
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_congestion = self._optimal_values[self._current_observation_index][
            self._tm_start_index + self._history_length]
        kl_value = self.optimizer_step(links_weights, tm, optimal_congestion)

        return kl_value

    def optimizer_step(self, links_weights, tm, optimal_value):
        kl_value = self._optimizer.step(links_weights, tm, optimal_value)
        return kl_value
