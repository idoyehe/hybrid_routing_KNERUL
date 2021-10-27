"""
Created on 14 Oct 2020
@author: Ido Yehezkel
"""
from gym import Env, spaces
from common.network_class import *
from common.RL_Envs.rl_env_consts import EnvConsts, ExtraData
from common.utils import load_dump_file
from common.topologies import topology_zoo_loader
from .optimizer_abstract import Optimizer_Abstract
import random


class RL_Env(Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 test_file=None,
                 history_length=None,
                 num_train_observations=None,
                 num_test_observations=None,
                 testing=False):

        self._NORM_FACTOR = -1

        self._episode_len = max_steps

        train_loaded_dict = load_dump_file(file_name=path_dumped)
        test_loaded_dict = load_dump_file(file_name=test_file)
        self._network = NetworkClass(topology_zoo_loader(url=train_loaded_dict["url"]))
        self._expected_congestion = train_loaded_dict["expected_congestion"]
        self._dst_mean_congestion = train_loaded_dict["dst_mean_congestion"]
        logger.info("Expected Congestion: {}".format(self._expected_congestion))
        logger.info("Dest Mean Congestion Result: {}".format(self._dst_mean_congestion))

        self._initial_weights = train_loaded_dict["initial_weights"]

        self._tms = train_loaded_dict["tms"]
        random.shuffle(self._tms)
        self._tms = self._tms[0:num_train_observations]

        self._tms_test = test_loaded_dict["tms"]
        random.shuffle(self._tms_test)
        self._tms_test = self._tms_test[0:num_test_observations]
        self._tm_type = train_loaded_dict["tms_type"]

        assert len(self._tms) == num_train_observations
        assert len(self._tms_test) == num_test_observations

        if "oblivious_routing" in train_loaded_dict.keys():
            self._oblivious_routing_per_edge = train_loaded_dict["oblivious_routing"]["per_edge"]
            self._oblivious_routing_per_flow = train_loaded_dict["oblivious_routing"]["per_flow"]
        else:
            self._oblivious_routing_per_edge = self._oblivious_routing_per_flow = None

        del train_loaded_dict, test_loaded_dict
        self._network = self._network
        self._g_name = self._network.get_title
        self._num_nodes = self._network.get_num_nodes
        self._num_edges = self._network.get_num_edges
        self._tm_start_index = 0
        self._current_observation_index = -1

        self._history_length = history_length  # number of each state history
        self._none_history = self._history_length == 0
        self._num_train_observations = num_train_observations  # number of different train seniors
        self._num_test_observations = num_test_observations  # number of different test seniors
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
        self._action_space = spaces.Box(low=EnvConsts.WEIGHT_LB, high=EnvConsts.WEIGHT_UB, shape=(self._num_edges,))

    def _sample_tm(self, tm_set):
        # we need to make the TM change slowly in time, currently it changes every step kind of drastically
        for idx, tuple_element in enumerate(tm_set):
            tm, opt = tuple_element
            yield tm, opt

    def _init_all_observations(self):
        def __create_episode(_episode_len, set_gen):
            _episode_tms = list()
            _episode_optimals = list()
            for _ in range(_episode_len):
                tm, opt = next(set_gen)
                _episode_tms.append(tm)
                _episode_optimals.append(opt)
            return _episode_tms, _episode_optimals

        def __create_observation(_num_observations, set_gen):
            _observations_episodes = list()
            _observations_episodes_optimals = list()
            for _ in range(_num_observations):
                _episode_tms, _episode_optimals = __create_episode(self._history_length + self._episode_len, set_gen)
                _observations_episodes.append(np.array(_episode_tms))
                _observations_episodes_optimals.append(np.array(_episode_optimals))
            return np.array(_observations_episodes), np.array(_observations_episodes_optimals)

        train_set_gen = self._sample_tm(self._tms)
        test_set_gen = self._sample_tm(self._tms_test)
        self._train_observations, self._opt_train_observations = __create_observation(self._num_train_observations, train_set_gen)
        self._test_observations, self._opt_test_observations = __create_observation(self._num_test_observations, test_set_gen)
        # self._validate_data()

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
    def get_num_test_observations(self) -> int:
        return self._num_test_observations

    @property
    def get_optimizer(self) -> Optimizer_Abstract:
        return self._optimizer

    @property
    def get_train_observations(self):
        return self._train_observations, self._opt_train_observations

    @property
    def get_expected_congestion(self):
        return self._expected_congestion

    @property
    def get_initial_weights(self):
        return self._initial_weights

    def set_train_observations(self, train_observations):
        self._train_observations = train_observations[0]
        self._opt_train_observations = train_observations[1]
