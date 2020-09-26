"""
Created on 24 Sep 2020
@author: Ido Yehezkel
"""

from ecmp_history import *


class ECMPHistorySimplifiedEnv(ECMPHistoryEnv):
    def __init__(self,
                 max_steps,
                 topo_customize,
                 bottleneck_th=0.2,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 train_histories_length=None,
                 test_histories_length=None,
                 testing=False):
        self._bottleneck_links_num = 0
        super().__init__(max_steps=max_steps, path_dumped=path_dumped, history_length=history_length,
                         history_action_type=history_action_type, train_histories_length=train_histories_length,
                         test_histories_length=test_histories_length, testing=testing)
        self._network = NetworkClass(topo=topology_zoo_loader(topo_customize)).get_g_directed
        self._static_weights = np.zeros(shape=(self._network.get_num_edges))
        self._dynamic_link_weights_map = None
        self._bottleneck_links_parsing(bottleneck_th)
        self._optimizer = WNumpyOptimizer(self._network)
        self._set_action_space()

    def _bottleneck_links_parsing(self, bottleneck_th):
        non_bottleneck_links = 0
        self._dynamic_link_weights_map = dict()
        for (src, dst, edge_dict) in self._network.edges.data():
            for key, val in edge_dict.items():
                edge_dict[key] = float(val)
            edge_id = self._network.get_edge2id()[(src, dst)]
            if edge_dict["std"] <= bottleneck_th:
                edge_dict["bottleneck"] = False
                non_bottleneck_links += 1
                self._static_weights[edge_id] = edge_dict["mean"]
            else:
                edge_dict["bottleneck"] = True
                self._dynamic_link_weights_map[self._bottleneck_links_num] = edge_id
                self._bottleneck_links_num += 1
        assert non_bottleneck_links + self._bottleneck_links_num == self._network.get_num_edges

    def _set_action_space(self):
        self._action_space = spaces.Box(low=0, high=np.inf, shape=(self._bottleneck_links_num,))

    def _process_action(self, action):
        action = super(ECMPHistorySimplifiedEnv, self)._process_action(action)
        weights = np.zeros(shape=(self._num_edges,))
        for action_index, weights_index in self._dynamic_link_weights_map.items():
            weights[weights_index] = action[action_index]
        assert self._static_weights.shape == weights.shape
        weights += self._static_weights
        return weights
