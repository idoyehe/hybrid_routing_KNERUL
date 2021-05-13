import numpy as np

from common.RL_Env.rl_env import *
from common.utils import error_bound
from common.RL_Env.optimizer_abstract import Optimizer_Abstract
from experiments.soft_min_smart_node_optimizer import SoftMinSmartNodesOptimizer


class RL_Env_BT(RL_Env):

    def __init__(self,
                 max_steps=1,
                 path_dumped=None,
                 history_length=0,
                 history_action_type=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):
        super(RL_Env_BT, self).__init__(max_steps=max_steps, path_dumped=path_dumped,
                                        history_length=history_length,
                                        history_action_type=history_action_type,
                                        num_train_observations=num_train_observations,
                                        num_test_observations=num_test_observations, testing=testing)

        assert isinstance(self._optimizer, Optimizer_Abstract)
        self._diagnostics = list()

    @property
    def diagnostics(self):
        return np.array(self._diagnostics)

    def _set_action_space(self):
        self._action_space = spaces.Box(low=0, high=np.inf, shape=(self._num_edges,))

    def step(self, action):
        info = dict()
        action = self._modify_action(action)

        total_congestion, cost_congestion_ratio, total_load_per_link, most_congested_arch = self._process_action_get_cost(action)
        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        # oblivious_value = self._oblivious_values[self._current_observation_index][self._tm_start_index + self._history_length]

        if self._testing:
            info["actions"] = np.array(action)
            info["load_per_link"] = np.array(total_load_per_link)

        del total_load_per_link
        info[ExtraData.REWARD_OVER_FUTURE] = cost_congestion_ratio
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()
        reward = cost_congestion_ratio * self._NORM_FACTOR
        done = self._is_terminal
        return observation, reward, done, info

    def _process_action_get_cost(self, links_weights):
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_congestion = self._optimal_values[self._current_observation_index][self._tm_start_index + self._history_length]
        total_congestion, max_congestion, total_load_per_arch, most_congested_arch = self.optimizer_step(links_weights, tm, optimal_congestion)

        cost_congestion_ratio = max_congestion / optimal_congestion

        if cost_congestion_ratio < 1.0:
            try:
                assert error_bound(cost_congestion_ratio, optimal_congestion, ERROR_BOUND)
            except Exception as _:
                logger.info("BUG!! Cost Congestion Ratio is {} not validate error bound!\n"
                            "Max Congestion: {}\nOptimal Congestion: {}".format(cost_congestion_ratio, max_congestion,
                                                                                optimal_congestion))

        cost_congestion_ratio = max(cost_congestion_ratio, 1.0)
        logger.debug("Cost  Congestion :{}".format(max_congestion))
        logger.debug("optimal  Congestion :{}".format(optimal_congestion))
        logger.debug("Congestion Ratio :{}".format(cost_congestion_ratio))

        return total_congestion, cost_congestion_ratio, total_load_per_arch, most_congested_arch

    def optimizer_step(self, links_weights, tm, optimal_value):
        total_congestion, max_congestion, \
        total_load_per_arch, most_congested_arch = self._optimizer.step(links_weights, tm, optimal_value)
        return total_congestion, max_congestion, total_load_per_arch, most_congested_arch

    def testing(self, _testing):
        super(RL_Env_BT, self).testing(_testing)
        self._optimizer = SoftMinSmartNodesOptimizer(self._network, testing=_testing)

    def set_network_smart_nodes_and_spr(self, smart_nodes, smart_nodes_spr):
        self._network.set_smart_nodes(smart_nodes)
        self._network.set__smart_nodes_spr(smart_nodes_spr)
