"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 14 Oct 2020
@by: Ido Yehezkel
"""

from rl_env import *
from Learning_to_Route.common.utils import error_bound
from optimizer import WNumpyOptimizer
from optimizer_refine import WNumpyOptimizer_Refine


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

        congestion_ratio, cost, optimal_congestion, routing_scheme = self._process_action_get_cost(links_weights)
        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        if self._testing:
            info["links_weights"] = np.array(links_weights)
            info["routing_scheme"] = np.array(routing_scheme)
        info[ExtraData.REWARD_OVER_FUTURE] = congestion_ratio
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()

        logger.debug("cost  Congestion :{}".format(cost))
        logger.debug("optimal  Congestion :{}".format(optimal_congestion))
        logger.debug("Congestion Ratio :{}".format(congestion_ratio))

        if not congestion_ratio >= 1.0:
            assert error_bound(cost, optimal_congestion, 5e-4)
            logger.info(
                "BUG!! congestion_ratio is {} not validate error bound!\nCost: {}\nOptimal: {}".format(congestion_ratio,
                                                                                                       cost,
                                                                                                       optimal_congestion))
            congestion_ratio = 1.0

        reward = congestion_ratio * self._NORM_FACTOR
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
        cost, routing_scheme = self.optimizer_step(links_weights, tm)
        congestion_ratio = cost / optimal_congestion
        assert congestion_ratio == cost / optimal_congestion
        return congestion_ratio, cost, optimal_congestion, routing_scheme

    def optimizer_step(self, links_weights, tm):
        routing_scheme = None
        if self._testing:
            cost, _, routing_scheme = self._optimizer.step(links_weights, tm)
        else:
            cost, _ = self._optimizer.step(links_weights, tm)
        return cost, routing_scheme

    def testing(self, _testing):
        super(RL_Env_History, self).testing(_testing)
        self._optimizer = WNumpyOptimizer_Refine(self._network)
