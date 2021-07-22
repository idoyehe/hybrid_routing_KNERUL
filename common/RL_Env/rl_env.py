"""
Created on 14 Oct 2020
@author: Ido Yehezkel
"""
import numpy as np
from gym import Env, spaces
from common.network_class import *
from common.RL_Env.rl_env_consts import HistoryConsts, ExtraData
from common.utils import load_dump_file
from common.topologies import topology_zoo_loader
import random

ERROR_BOUND = 1e-3


class RL_Env(Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 history_length=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):

        self._NORM_FACTOR = -1

        self._episode_len = max_steps

        loaded_dict = load_dump_file(file_name=path_dumped)
        self._network = NetworkClass(topology_zoo_loader(url=loaded_dict["url"]))
        self._tms = loaded_dict["tms"]
        self._tm_type = loaded_dict["tms_type"]
        if "oblivious_routing" in loaded_dict.keys():
            self._oblivious_routing_per_edge = loaded_dict["oblivious_routing"]["per_edge"]
            self._oblivious_routing_per_flow = loaded_dict["oblivious_routing"]["per_flow"]
        else:
            self._oblivious_routing_per_edge = self._oblivious_routing_per_flow = None
        self._network = self._network
        self._g_name = self._network.get_title
        self._num_nodes = self._network.get_num_nodes
        self._num_edges = self._network.get_num_edges
        self._tm_start_index = 0
        self._current_observation_index = -1

        self._history_length = history_length  # number of each state history
        self._none_history = self._history_length == 0
        self._num_train_observations = num_train_observations  # number of different train seniors per sparsity
        self._num_test_observations = num_test_observations  # number of different test seniors per sparsity
        self._observation_space = None
        self._action_space = None
        self._optimizer = None

        # init states spaces and action space
        self._set_action_space()
        self._set_observation_space()
        self._init_all_observations()
        self._testing = None
        self.testing(testing)

    def get_num_steps(self):
        return self._episode_len

    def get_num_steps(self):
        return self._episode_len

    def _set_observation_space(self):
        if self._none_history:
            self._observation_space = spaces.Box(low=0.0, high=np.inf, shape=(self._num_nodes, self._num_nodes))
        else:
            self._observation_space = spaces.Box(low=0.0, high=np.inf, shape=(self._history_length, self._num_nodes, self._num_nodes))

    def _set_action_space(self):
        self._action_space = spaces.Box(low=HistoryConsts.EPSILON, high=np.inf, shape=(self._num_edges,))

    def _sample_tm(self):
        # we need to make the TM change slowly in time, currently it changes every step kind of drastically
        tuple_element = random.choice(self._tms)
        oblv = None
        if len(tuple_element) == 3:
            tm, opt, oblv = tuple_element
        else:
            tm, opt = tuple_element
        return tm, opt, oblv

    def _init_all_observations(self):
        def __create_episode(_episode_len):
            _episode_tms = list()
            _episode_optimals = list()
            _episode_oblivious = list()
            for _ in range(_episode_len):
                tm, opt, oblv = self._sample_tm()
                _episode_tms.append(tm)
                _episode_optimals.append(opt)
                _episode_oblivious.append(oblv)
            return _episode_tms, _episode_optimals, _episode_oblivious

        def __create_observation(_num_observations):
            _observations_episodes = list()
            _observations_episodes_optimals = list()
            _observations_episodes_oblivious = list()
            for _ in range(_num_observations):
                _episode_tms, _episode_optimals, _episode_oblivious = __create_episode(self._history_length + self._episode_len)
                _observations_episodes.append(np.array(_episode_tms))
                _observations_episodes_optimals.append(np.array(_episode_optimals))
                _observations_episodes_oblivious.append(np.array(_episode_oblivious))
            return np.array(_observations_episodes), np.array(_observations_episodes_optimals), np.array(_observations_episodes_oblivious)

        self._train_observations, self._opt_train_observations, self._oblv_train_observations = __create_observation(self._num_train_observations)
        self._test_observations, self._opt_test_observations, self._oblv_test_observations = __create_observation(self._num_test_observations)
        if not self._none_history:
            self._validate_data()

    def _validate_data(self):
        is_equal_train = np.zeros((self._num_train_observations, self._num_train_observations))
        is_equal_test = np.zeros((self._num_test_observations, self._num_test_observations))
        is_equal_train_test = np.zeros((self._num_train_observations, self._num_test_observations))

        for i in range(self._num_train_observations):
            for j in range(i + 1, self._num_train_observations):
                is_equal_train[i, j] = np.array_equal(self._train_observations[i].flatten(),
                                                      self._train_observations[j].flatten())

        for i in range(self._num_test_observations):
            for j in range(i + 1, self._num_test_observations):
                is_equal_test[i, j] = np.array_equal(self._test_observations[i].flatten(),
                                                     self._test_observations[j].flatten())

        for i in range(self._num_train_observations):
            for j in range(self._num_test_observations):
                is_equal_train_test[i, j] = np.array_equal(self._train_observations[i].flatten(),
                                                           self._test_observations[j].flatten())

        assert np.sum(is_equal_train) == 0.0
        assert np.sum(is_equal_test) == 0.0
        assert np.sum(is_equal_train_test) == 0.0

    def set_data_source(self):
        self._observations_tms = self._test_observations if self._testing else self._train_observations
        self._observations_length = self._num_test_observations if self._testing else self._num_train_observations
        self._optimal_values = self._opt_test_observations if self._testing else self._opt_train_observations
        self._oblivious_values = self._oblv_test_observations if self._testing else self._oblv_train_observations

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def testing(self, _testing):
        self._testing = _testing
        self.set_data_source()

    def _get_observation(self, by_env_reset: bool = False):
        if self._none_history:
            if by_env_reset:
                self._current_history = np.zeros(shape=(self._num_nodes, self._num_nodes))
            else:
                self._current_history = self._observations_tms[self._current_observation_index][0]
        else:
            self._current_history = np.stack(
                self._observations_tms[self._current_observation_index][self._tm_start_index:self._tm_start_index + self._history_length])
        return self._current_history

    def render(self, mode='human'):
        pass

    def reset(self):
        self._tm_start_index = 0
        self._current_observation_index = (self._current_observation_index + 1) % self._observations_length
        return self._get_observation(True)

    def step(self, action):
        pass

    @property
    def get_network(self) -> NetworkClass:
        return self._network

    @property
    def get_optimizer(self):
        return self._optimizer
