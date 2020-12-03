"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 14 Oct 2020
@by: Ido Yehezkel
"""

from rl_env import *
from Learning_to_Route.common.utils import error_bound
from optimizer import WNumpyOptimizer

ERROR_BOUND = 5e-4


class RL_Env_History(RL_Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):
        super(RL_Env_History, self).__init__(max_steps=max_steps, path_dumped=path_dumped,
                                             history_length=history_length,
                                             history_action_type=history_action_type,
                                             num_train_observations=num_train_observations,
                                             num_test_observations=num_test_observations, testing=testing)

        self._num_edges = self._network.get_num_edges
        self._optimizer = WNumpyOptimizer(self._network)
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

        cost_congestion_ratio, total_load_per_arch, most_congested_arch = self._process_action_get_cost(links_weights)
        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        if self._testing:
            info["links_weights"] = np.array(links_weights)
            info["load_per_link"] = np.array(total_load_per_arch)
            info["most_congested_link"] = self._network.get_id2edge()[most_congested_arch]
        del total_load_per_arch
        del links_weights
        info[ExtraData.REWARD_OVER_FUTURE] = cost_congestion_ratio
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()

        reward = cost_congestion_ratio * self._NORM_FACTOR
        done = self._is_terminal

        return observation, reward, done, info

    def reset(self):
        self._tm_start_index = 0
        self._current_observation_index = (self._current_observation_index + 1) % self._observations_length
        return self._get_observation()

    def _process_action_get_cost(self, links_weights):
        global ERROR_BOUND
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_congestion = self._optimal_values[self._current_observation_index][
            self._tm_start_index + self._history_length]
        max_congestion, total_load_per_arch, most_congested_arch = self.optimizer_step(links_weights, tm)

        cost_congestion_ratio = max_congestion / optimal_congestion

        if cost_congestion_ratio < 1.0:
            assert error_bound(cost_congestion_ratio, optimal_congestion, ERROR_BOUND)
            logger.info("BUG!! Cost Congestion Ratio is {} not validate error bound!\n"
                        "Max Congestion: {}\nOptimal Congestion: {}"
                        .format(cost_congestion_ratio, max_congestion, optimal_congestion))

        cost_congestion_ratio = max(cost_congestion_ratio, 1.0)
        logger.debug("Cost  Congestion :{}".format(max_congestion))
        logger.debug("optimal  Congestion :{}".format(optimal_congestion))
        logger.debug("Congestion Ratio :{}".format(cost_congestion_ratio))

        return cost_congestion_ratio, total_load_per_arch, most_congested_arch

    def optimizer_step(self, links_weights, tm):
        max_congestion, total_load_per_arch, most_congested_arch = self._optimizer.step(links_weights, tm)
        return max_congestion, total_load_per_arch, most_congested_arch
