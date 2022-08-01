"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 14 Oct 2020
@by: Ido Yehezkel
"""

from common.RL_Envs.rl_env import *
from common.utils import error_bound
from soft_min_optimizer import SoftMinOptimizer


class RL_Env_History(RL_Env):

    def __init__(self,
                 path_dumped=None,
                 test_file=None,
                 history_length=None,
                 num_train_episodes=None,
                 num_test_episodes=None,
                 testing=False):
        super(RL_Env_History, self).__init__(path_dumped=path_dumped, test_file=test_file,
                                             history_length=history_length,
                                             num_train_episodes=num_train_episodes,
                                             num_test_episodes=num_test_episodes, testing=testing)

        assert isinstance(self._optimizer, SoftMinOptimizer)
        self._diagnostics = list()

    @property
    def diagnostics(self):
        return np.array(self._diagnostics)

    def step(self, links_weights):
        info = dict()

        cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._process_action_get_cost(links_weights)

        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        info[ExtraData.REWARD_OVER_FUTURE] = cost_congestion_ratio
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()

        reward = cost_congestion_ratio * self._NORM_FACTOR
        done = self._is_terminal
        del links_weights
        del cost_congestion_ratio
        del most_congested_link
        del flows_to_dest_per_node
        del total_congestion_per_link

        return observation, reward, done, info

    def _process_action_get_cost(self, links_weights):
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_val = self._optimal_values[self._current_observation_index][self._tm_start_index + self._history_length]
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self.optimizer_step(links_weights, tm, optimal_val)

        if self._testing:
            cost_congestion_ratio = max_congestion
        else:
            cost_congestion_ratio = max_congestion / optimal_val

            if cost_congestion_ratio < 1.0:
                try:
                    assert error_bound(max_congestion, optimal_val)
                except Exception as _:
                    logger.info("BUG!! Cost Congestion Ratio is {} not validate error bound!\n"
                                "Max Congestion: {}\nOptimal Congestion: {}".format(cost_congestion_ratio, max_congestion, optimal_val))
            cost_congestion_ratio = max(cost_congestion_ratio, 1.0)


        logger.debug("SoftMin  Congestion :{}".format(max_congestion))
        logger.debug("Optimal  Congestion :{}".format(optimal_val))
        logger.debug("Congestion Ratio :{}".format(cost_congestion_ratio))

        return cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def optimizer_step(self, links_weights, tm, optimal_value):
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._optimizer.step(links_weights, tm, optimal_value)
        return max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def testing(self, _testing):
        super(RL_Env_History, self).testing(_testing)
        self._optimizer = SoftMinOptimizer(self._network, EnvConsts.SOFTMIN_GAMMA, testing=_testing)
