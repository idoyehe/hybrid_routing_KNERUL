from common.RL_Envs.rl_env import *
from common.utils import error_bound
from common.RL_Envs.optimizer_abstract import Optimizer_Abstract
from Smart_Nodes_Routing.rl_env.soft_min_smart_node_algebraic_optimizer import SoftMinSmartNodesOptimizer
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
from common.static_routing.optimal_load_balancing import *

class RL_Smart_Nodes(RL_Env):

    def __init__(self,
                 max_steps=1,
                 path_dumped=None,
                 test_file=None,
                 history_length=0,
                 num_train_observations=None,
                 num_test_observations=None,
                 softMin_gamma=EnvConsts.SOFTMIN_GAMMA,
                 action_weight_lb=EnvConsts.WEIGHT_LB,
                 action_weight_ub=EnvConsts.WEIGHT_UB,
                 testing=False):

        self._softMin_gamma = softMin_gamma
        self._action_weight_lb = action_weight_lb
        self._action_weight_ub = action_weight_ub

        super(RL_Smart_Nodes, self).__init__(max_steps=max_steps, path_dumped=path_dumped, test_file=test_file,
                                             history_length=history_length,
                                             num_train_observations=num_train_observations,
                                             num_test_observations=num_test_observations, testing=testing)

        assert isinstance(self._optimizer, Optimizer_Abstract)
        self._diagnostics = list()

    def _set_action_space(self):
        self._action_space = spaces.Box(low=self._action_weight_lb, high=self._action_weight_ub, shape=(self._num_edges,))

    @property
    def diagnostics(self):
        return np.array(self._diagnostics)

    def step(self, links_weights):
        cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._process_action_get_cost(links_weights)
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
        oblivious_congestion = self._oblivious_values[self._current_observation_index][self._tm_start_index + self._history_length]
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = self.optimizer_step(
            links_weights, tm, optimal_congestion)

        if self._testing:
            cost_congestion_ratio = max_congestion
        else:
            cost_congestion_ratio = max_congestion / optimal_congestion

            if cost_congestion_ratio < 1.0:
                try:
                    assert error_bound(max_congestion, optimal_congestion, ERROR_BOUND)
                except Exception as _:
                    logger.info("BUG!! Cost Congestion Ratio is {} not validate error bound!\n"
                                "Max Congestion: {}\nOptimal Congestion: {}".format(cost_congestion_ratio, max_congestion,
                                                                                    optimal_congestion))

            cost_congestion_ratio = max(cost_congestion_ratio, 1.0)

        logger.debug("optimal Congestion :{}".format(optimal_congestion))
        logger.debug("Cost Congestion :{}".format(max_congestion))
        logger.debug("oblivious Congestion :{}".format(oblivious_congestion))
        logger.debug("Congestion Ratio :{}".format(cost_congestion_ratio))

        return cost_congestion_ratio, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def optimizer_step(self, links_weights, tm, optimal_value):
        max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link = \
            self._optimizer.step(links_weights, tm, optimal_value)
        return max_congestion, most_congested_link, flows_to_dest_per_node, total_congestion_per_link, total_load_per_link

    def testing(self, _testing):
        super(RL_Smart_Nodes, self).testing(_testing)
        # self._optimizer = SoftMinSmartNodesOptimizer(self._network, self._softMin_gamma, testing=_testing)
        self._optimizer = PEFTOptimizer(self._network, testing=_testing)

    def set_network_smart_nodes_and_spr(self, smart_nodes, smart_nodes_spr):
        self._network.set_smart_nodes(smart_nodes)
        self._network.set__smart_nodes_spr(smart_nodes_spr)
        self._optimizer = SoftMinSmartNodesOptimizer(self._network, self._softMin_gamma, testing=self._testing)
