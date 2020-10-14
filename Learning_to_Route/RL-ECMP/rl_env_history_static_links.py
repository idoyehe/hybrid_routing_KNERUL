"""
Created on 24 Sep 2020
@author: Ido Yehezkel
"""

from rl_env import *
from rl_env_history import *


class RL_Env_History_Static_Weights(RL_Env_History):
    def __init__(self,
                 max_steps,
                 topo_customize,
                 dynamic_link_std_thresh=0.2,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):
        self._dynamic_links_num = 0
        super(RL_Env_History_Static_Weights, self).__init__(max_steps=max_steps, path_dumped=path_dumped,
                                                            history_length=history_length,
                                                            history_action_type=history_action_type,
                                                            num_train_observations=num_train_observations,
                                                            num_test_observations=num_test_observations,
                                                            testing=testing)


        self._network = NetworkClass(topo=topology_zoo_loader(topo_customize))
        self._network = self._network.get_g_directed
        self._g_name = self._network.get_name
        self._num_nodes = self._network.get_num_nodes
        self._num_edges = self._network.get_num_edges
        self._optimizer = WNumpyOptimizer(self._network)

        self._static_weights = np.zeros(shape=(self._num_edges,))
        self._dynamic_link_weights_map = None
        self._dynamic_static_links_parsing(dynamic_link_std_thresh)
        self._set_action_space()

    def _dynamic_static_links_parsing(self, _dynamic_link_std_thresh):
        non_bottleneck_links = 0
        self._dynamic_link_weights_map = dict()
        for (src, dst, edge_dict) in self._network.edges.data():
            for key, val in edge_dict.items():
                edge_dict[key] = float(val)
            edge_id = self._network.get_edge2id()[(src, dst)]
            if edge_dict[HistoryConsts.STD_LINK_VALUE] <= _dynamic_link_std_thresh:
                edge_dict[HistoryConsts.DYNAMIC_LINK] = False
                non_bottleneck_links += 1
                self._static_weights[edge_id] = edge_dict[HistoryConsts.STD_MEAN_VALUE]
            else:
                edge_dict[HistoryConsts.DYNAMIC_LINK] = True
                self._dynamic_link_weights_map[self._dynamic_links_num] = edge_id
                self._dynamic_links_num += 1
        assert non_bottleneck_links + self._dynamic_links_num == self._network.get_num_edges

    def _set_action_space(self):
        self._action_space = spaces.Box(low=0, high=np.inf, shape=(self._dynamic_links_num,))

    def _modify_action(self, action):
        action = super(RL_Env_History_Static_Weights, self)._modify_action(action)
        dynamic_weights = np.zeros(shape=(self._num_edges,))
        for dynamic_link_index, dynamic_link_weight in enumerate(action):
            edge_id = self._dynamic_link_weights_map[dynamic_link_index]
            assert self._static_weights[edge_id] == 0.0
            dynamic_weights[edge_id] = dynamic_link_weight
        assert self._static_weights.shape == dynamic_weights.shape
        action = dynamic_weights + self._static_weights
        return action
