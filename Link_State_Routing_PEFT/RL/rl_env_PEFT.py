from common.RL_Envs.rl_env import *
from common.utils import error_bound
from common.RL_Envs.optimizer_abstract import Optimizer_Abstract
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer


class RL_Env_PEFT(RL_Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 num_train_episodes=None,
                 num_test_episodes=None,
                 testing=False):
        super(RL_Env_PEFT, self).__init__(max_steps=max_steps, path_dumped=path_dumped,
                                          history_length=history_length,
                                          history_action_type=history_action_type,
                                          num_train_episodes=num_train_episodes,
                                          num_test_episodes=num_test_episodes, testing=testing)

        self._num_edges = self._network.get_num_edges
        assert isinstance(self._optimizer, Optimizer_Abstract)
        self._set_action_space()

        self._diagnostics = list()

    @property
    def diagnostics(self):
        return np.array(self._diagnostics)

    def _set_action_space(self):
        self._action_space = spaces.Box(low=2, high=np.inf, shape=(self._num_edges,))

    def _set_observation_space(self):
        self._observation_space = spaces.Box(low=0.0, high=np.inf,
                                             shape=(self._history_length, self._num_nodes, self._num_nodes),
                                             dtype=np.float64)

    def step(self, action):
        info = dict()
        links_weights = self._modify_action(action)

        cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._process_action_get_cost(links_weights)
        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        oblivious_value = self._oblivious_values[self._current_observation_index][
            self._tm_start_index + self._history_length]

        if self._testing:
            info["links_weights"] = np.array(links_weights)
            info["load_per_link"] = np.array(total_load_per_link)
            info["most_congested_link"] = most_congested_link

        del total_load_per_link
        del links_weights
        info[ExtraData.REWARD_OVER_FUTURE] = cost_congestion_ratio
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()

        reward = cost_congestion_ratio * self._NORM_FACTOR
        done = self._is_terminal

        return observation, reward, done, info

    def _process_action_get_cost(self, links_weights):
        global ERROR_BOUND
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_congestion = self._optimal_values[self._current_observation_index][
            self._tm_start_index + self._history_length]
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self.optimizer_step(links_weights, tm, optimal_congestion)

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

        return cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def optimizer_step(self, links_weights, tm, optimal_value):
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._optimizer.step(links_weights, tm, optimal_value)
        return max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def testing(self, _testing):
        super(RL_Env_PEFT, self).testing(_testing)
        self._optimizer = PEFTOptimizer(self._network, self._oblivious_routing_per_edge, testing=_testing)
