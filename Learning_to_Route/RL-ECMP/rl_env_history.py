"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 14 Oct 2020
@by: Ido Yehezkel
"""

from common.RL_Env.rl_env import *
from common.utils import error_bound
from soft_min_optimizer import SoftMinOptimizer


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
        assert isinstance(self._optimizer, SoftMinOptimizer)
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

        cost_congestion_ratio, most_congested_link, total_congestion, total_congestion_per_link, \
        total_load_per_link = self._process_action_get_cost(links_weights)
        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        oblivious_value = self._oblivious_values[self._current_observation_index][
            self._tm_start_index + self._history_length]

        if self._testing:
            info[ExtraData.LINK_WEIGHTS] = np.array(links_weights)
            info[ExtraData.LOAD_PER_LINK] = np.array(total_load_per_link)
            info[ExtraData.MOST_CONGESTED_LINK] = most_congested_link
            info[ExtraData.VS_OBLIVIOUS_DATA] = self._optimizer.vs_oblivious_data

        info[ExtraData.REWARD_OVER_FUTURE] = cost_congestion_ratio
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()

        reward = cost_congestion_ratio * self._NORM_FACTOR
        done = self._is_terminal
        del links_weights
        del cost_congestion_ratio
        del most_congested_link
        del total_congestion
        del total_congestion_per_link

        return observation, reward, done, info

    def _process_action_get_cost(self, links_weights):
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_val = self._optimal_values[self._current_observation_index][self._tm_start_index + self._history_length]
        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
            self.optimizer_step(links_weights, tm, optimal_val)

        cost_congestion_ratio = max_congestion / optimal_val

        if cost_congestion_ratio < 1.0:
            try:
                assert error_bound(cost_congestion_ratio, optimal_val)
            except Exception as _:
                logger.info("BUG!! Cost Congestion Ratio is {} not validate error bound!\n"
                            "Max Congestion: {}\nOptimal Congestion: {}".format(cost_congestion_ratio, max_congestion,
                                                                                optimal_val))

        cost_congestion_ratio = max(cost_congestion_ratio, 1.0)
        logger.debug("SoftMin  Congestion :{}".format(max_congestion))
        logger.debug("Optimal  Congestion :{}".format(optimal_val))
        logger.debug("Congestion Ratio :{}".format(cost_congestion_ratio))

        return cost_congestion_ratio, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    def optimizer_step(self, links_weights, tm, optimal_value):
        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
            self._optimizer.step(links_weights, tm, optimal_value)
        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    def testing(self, _testing):
        super(RL_Env_History, self).testing(_testing)
        self._optimizer = SoftMinOptimizer(self._network, self._oblivious_routing_per_edge, testing=_testing)
