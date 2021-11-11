from common.RL_Envs.rl_env import *
from common.RL_Envs.optimizer_abstract import Optimizer_Abstract
from Smart_Nodes_Routing.rl_env.smart_node_algebraic_optimizer import SmartNodesOptimizer
from common.static_routing.optimal_load_balancing import *


class RL_Smart_Nodes(RL_Env):

    def __init__(self,
                 max_steps=1,
                 path_dumped=None,
                 test_file=None,
                 history_length=0,
                 num_train_observations=None,
                 num_test_observations=None,
                 weights_factor=EnvConsts.WEIGHTS_FACTOR,
                 action_weight_lb=EnvConsts.WEIGHT_LB,
                 action_weight_ub=EnvConsts.WEIGHT_UB,
                 testing=False):

        self._weight_factor = weights_factor
        self._action_weight_lb = action_weight_lb
        self._action_weight_ub = action_weight_ub

        super(RL_Smart_Nodes, self).__init__(max_steps=max_steps, path_dumped=path_dumped, test_file=test_file,
                                             history_length=history_length,
                                             num_train_observations=num_train_observations,
                                             num_test_observations=num_test_observations, testing=testing)

        assert isinstance(self._optimizer, Optimizer_Abstract)
        self._diagnostics = list()
        if logger.level == logging.DEBUG:
            self.softMin_initial_expected_congestion()


    def _set_action_space(self):
        self._action_space = spaces.Box(low=self._action_weight_lb, high=self._action_weight_ub, shape=(self._num_edges,), dtype=np.float64)

    @property
    def diagnostics(self):
        return np.array(self._diagnostics)

    def step(self, links_weights):
        links_weights *= self._weight_factor
        cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, \
        total_congestion_per_link, total_load_per_link = self._process_action_get_cost(links_weights)

        self._is_terminal = self._tm_start_index + 1 == self._episode_len

        info = dict()
        if self._testing:
            info[ExtraData.LOAD_PER_LINK] = np.array(total_load_per_link)
            info[ExtraData.MOST_CONGESTED_LINK] = most_congested_link
            info[ExtraData.CONGESTION_PER_LINK] = np.array(total_congestion_per_link)

        del total_load_per_link
        self._diagnostics.append(info)

        self._tm_start_index += 1
        observation = self._get_observation()
        reward = cost_congestion_ratio * self._NORM_FACTOR
        done = self._is_terminal
        return observation, reward, done, info

    def _process_action_get_cost(self, links_weights):
        tm = self._observations_tms[self._current_observation_index][self._tm_start_index + self._history_length]
        optimal_congestion = self._optimal_values[self._current_observation_index][self._tm_start_index + self._history_length]
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = self.optimizer_step(
            links_weights, tm, optimal_congestion)

        cost_congestion_ratio = max_congestion

        logger.debug("optimal Congestion :{}".format(optimal_congestion))
        logger.debug("Max Congestion :{}".format(max_congestion))
        logger.debug("Congestion Ratio :{}".format(max_congestion/optimal_congestion))

        return cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def optimizer_step(self, links_weights, tm, optimal_value):
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._optimizer.step(links_weights, tm, optimal_value)
        return max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def testing(self, _testing):
        super(RL_Smart_Nodes, self).testing(_testing)
        self._optimizer = SmartNodesOptimizer(self._network, testing=_testing)

    def set_network_smart_nodes_and_spr(self, smart_nodes, smart_nodes_spr):
        self._network.set_smart_nodes(smart_nodes)
        self._network.set__smart_nodes_spr(smart_nodes_spr)
        self._optimizer = SmartNodesOptimizer(self._network, testing=self._testing)

    def softMin_initial_expected_congestion(self):
        dst_splitting_ratios = self._optimizer.calculating_destination_based_spr(self._initial_weights)
        a = np.mean([super(SmartNodesOptimizer,self._optimizer)._calculating_traffic_distribution(dst_splitting_ratios, tm[0])[0] for tm in self._test_observations])
        logger.info("SoftMin Initial Expected Congestion: {}".format(a))
