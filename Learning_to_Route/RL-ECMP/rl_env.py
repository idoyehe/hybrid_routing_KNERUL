"""
Created on 14 Oct 2020
@author: Ido Yehezkel
"""
from gym import Env, spaces
from common.network_class import *
from common.rl_env_consts import HistoryConsts, ExtraData
from static_routing.generating_tms_dumps import load_dump_file
from common.topologies import topology_zoo_loader
import random

ERROR_BOUND = 1e-3


class RL_Env(Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):

        self._NORM_FACTOR = -1

        self._episode_len = max_steps

        loaded_dict = load_dump_file(file_name=path_dumped)
        self._network = NetworkClass(
            topology_zoo_loader(url=loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
        self._tms = loaded_dict["tms"]
        self._tm_type = loaded_dict["tms_type"]
        self._tm_sparsity_list = loaded_dict["tms_sparsity"]  # percentage of participating pairs, assumed to be a list

        self._network = self._network.get_g_directed
        self._g_name = self._network.get_name
        self._num_nodes = self._network.get_num_nodes
        self._tm_start_index = 0
        self._current_observation_index = -1

        self._history_length = history_length  # number of each state history
        self._none_history = self._history_length == 0
        if self._none_history:
            self._history_length = 1
        self._history_action_type = history_action_type
        self._num_train_observations = num_train_observations  # number of different train seniors per sparsity
        self._num_test_observations = num_test_observations  # number of different test seniors per sparsity
        self._observation_space = None
        self._action_space = None

        # init states spaces and action space
        self._set_observation_space()
        self._init_all_observations()
        self._testing = None
        self.testing(testing)

    def get_num_steps(self):
        return self._episode_len

    def get_num_steps(self):
        return self._episode_len

    def _set_observation_space(self):
        self._observation_space = spaces.Box(low=0.0, high=np.inf,
                                             shape=(self._history_length, self._num_nodes, self._num_nodes))

    def _set_action_space(self):
        pass

    def _sample_tm(self):
        # we need to make the TM change slowly in time, currently it changes every step kind of drastically
        tm, opt = random.choice(self._tms)
        return tm, opt

    def _init_all_observations(self):
        def __create_episode(_episode_len):
            _episode_tms = list()
            _episode_optimals = list()
            for _ in range(_episode_len):
                tm, opt = self._sample_tm()
                _episode_tms.append(tm)
                _episode_optimals.append(opt)
            return _episode_tms, _episode_optimals

        def __create_observation(_num_observations):
            _observations_episodes = list()
            _observations_episodes_optimals = list()
            for _ in range(_num_observations):
                _episode_tms, _episode_optimals = __create_episode(self._history_length + self._episode_len)
                _observations_episodes.append(np.array(_episode_tms))
                _observations_episodes_optimals.append(np.array(_episode_optimals))
            return np.array(_observations_episodes), np.array(_observations_episodes_optimals)

        def fix_none_history_episode(observations):
            for ep in observations:
                ep[0] *= 0
            return observations

        self._train_observations, self._opt_train_observations = __create_observation(self._num_train_observations)
        self._test_observations, self._opt_test_observations = __create_observation(self._num_test_observations)

        if self._none_history:
            fix_none_history_episode(self._train_observations)
            fix_none_history_episode(self._opt_train_observations)
            fix_none_history_episode(self._test_observations)
            fix_none_history_episode(self._opt_test_observations)
        else:
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

    def _process_action(self, action):
        if self._history_action_type == HistoryConsts.ACTION_W_EPSILON:
            action[action <= 0] = HistoryConsts.EPSILON
        elif self._history_action_type == HistoryConsts.ACTION_W_INFTY:
            action[action <= 0] = HistoryConsts.INFTY
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def testing(self, _testing):
        self._testing = _testing
        self.set_data_source()

    def _get_observation(self):
        self._current_history = np.stack(
            self._observations_tms[self._current_observation_index][
            self._tm_start_index:self._tm_start_index + self._history_length])
        return self._current_history

    def _modify_action(self, action):
        if self._history_action_type == HistoryConsts.ACTION_W_EPSILON:
            action[action <= 0] = HistoryConsts.EPSILON
        elif self._history_action_type == HistoryConsts.ACTION_W_INFTY:
            action[action <= 0] = HistoryConsts.INFTY
        return action

    def render(self, mode='human'):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass
