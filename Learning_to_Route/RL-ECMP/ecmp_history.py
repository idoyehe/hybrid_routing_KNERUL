"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 26/04/2020
@by: Ido Yehezkel
"""
from Learning_to_Route.common.utils import error_bound
from gym import Env, spaces
from common.network_class import *
from optimizer import WNumpyOptimizer
from common.rl_env_consts import HistoryConsts, ExtraData
from static_routing.generating_tms_dumps import load_dump_file
from common.topologies import topology_zoo_loader
import random


class ECMPHistoryEnv(Env):

    def __init__(self,
                 max_steps,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 train_histories_length=None,
                 test_histories_length=None,
                 testing=False):

        self._episode_len = max_steps

        loaded_dict = load_dump_file(file_name=path_dumped)
        self._network = NetworkClass(
            topology_zoo_loader(url=loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
        self._tms = loaded_dict["tms"]
        self._tm_type = loaded_dict["tms_type"]
        self._tm_sparsity_list = loaded_dict["tms_sparsity"]  # percentage of participating pairs, assumed to be a list

        self._network = self._network.get_g_directed
        self._g_name = self._network.get_name
        self._num_edges = self._network.get_num_edges
        self._num_nodes = self._network.get_num_nodes
        self._all_pairs = self._network.get_all_pairs()
        self._history_start_id = 0
        self._current_history_index = -1
        self._optimizer = WNumpyOptimizer(self._network)

        self._history_len = history_length  # number of each state history
        self._history_action_type = history_action_type
        self._num_train_histories = train_histories_length  # number of different train seniors per sparsity
        self._num_test_histories = test_histories_length  # number of different test seniors per sparsity

        # init random placeholders, so we could refer to those in test function
        self._random_train = {}
        self._random_train_res = []
        self._random_train_std = []

        self._random_test = {}
        self._random_test_res = []
        self._random_test_std = []
        self._testing = testing

        # init states spaces and action space
        self._set_observation_space()
        self._set_action_space()
        self._init_all_observations()

        self._all_rewards = []
        self.diagnostics = []

    def get_num_steps(self):
        return self._episode_len

    def _set_observation_space(self):
        self._observation_space = spaces.Box(low=0.0, high=np.inf,
                                             shape=(self._history_len, self._num_nodes, self._num_nodes))

    def _set_action_space(self):
        self._action_space = spaces.Box(low=0, high=np.inf, shape=(self._num_edges,))

    def _sample_tm(self):
        # we need to make the TM change slowly in time, currently it changes every step kind of drastically
        tm, opt = random.choice(self._tms)
        return tm, opt

    def _init_all_observations(self):
        self._train_observations = []
        self._test_observations = []
        self._opt_train_observations = []
        self._opt_test_observations = []

        for _ in range(self._num_train_histories):
            train_episode = list()
            train_episode_optimal = list()
            for _ in range(self._history_len + self._episode_len):
                tm, opt = self._sample_tm()
                train_episode.append(tm)
                train_episode_optimal.append(opt)
            self._train_observations.append(train_episode)
            self._opt_train_observations.append(train_episode_optimal)

        for _ in range(self._num_test_histories):
            test_episode = list()
            test_episode_optimal = list()
            for _ in range(self._history_len + self._episode_len):
                tm, opt = self._sample_tm()
                test_episode.append(tm)
                test_episode_optimal.append(opt)
            self._test_observations.append(test_episode)
            self._opt_test_observations.append(test_episode_optimal)

        self._actual_num_train_histories = len(self._train_observations)
        self._actual_num_test_histories = len(self._test_observations)

        self._validate_data()

        self.test(self._testing)

    def _validate_data(self):
        is_equal_train = np.zeros((self._actual_num_train_histories, self._actual_num_train_histories))
        is_equal_test = np.zeros((self._actual_num_test_histories, self._actual_num_test_histories))
        is_equal_train_test = np.zeros((self._actual_num_train_histories, self._actual_num_test_histories))

        for i in range(self._actual_num_train_histories):
            for j in range(self._actual_num_train_histories):
                if i == j:
                    continue
                is_equal_train[i, j] = np.array_equal(self._train_observations[i], self._train_observations[j])

        for i in range(self._actual_num_test_histories):
            for j in range(self._actual_num_test_histories):
                if i == j: continue
                is_equal_test[i, j] = np.array_equal(self._test_observations[i], self._test_observations[j])

        for i in range(self._actual_num_train_histories):
            for j in range(self._actual_num_test_histories):
                is_equal_train_test[i, j] = np.array_equal(self._train_observations[i], self._test_observations[j])

        assert np.sum(is_equal_train) == 0.0
        assert np.sum(is_equal_test) == 0.0
        assert np.sum(is_equal_train_test) == 0.0

    def test(self, testing):
        self._observations = self._test_observations if testing else self._train_observations
        self._num_sequences = self._actual_num_test_histories if testing else self._actual_num_train_histories
        self._opt_res = self._opt_test_observations if testing else self._opt_train_observations

    def _process_action(self, action):
        if self._history_action_type == HistoryConsts.ACTION_W_EPSILON:
            action[action <= 0] = HistoryConsts.EPSILON
        elif self._history_action_type == HistoryConsts.ACTION_W_INFTY:
            action[action <= 0] = HistoryConsts.INFTY
        return action

    @property
    def observation_space(self):
        return self._observation_space

    def testing(self, _testing):
        self._testing = _testing

    @property
    def action_space(self):
        return self._action_space

    def _get_observation(self):
        self._current_history = np.stack(
            self._observations[self._current_history_index][
            self._history_start_id:self._history_start_id + self._history_len])
        return self._current_history

    def step(self, action):
        links_weights = self._process_action(action)

        cost = self._get_reward(links_weights)
        self._is_terminal = self._history_start_id + 1 == self._episode_len

        norm_factor = -1

        env_data = {}
        env_data["links_weights"] = links_weights
        optimal_congestion = self._opt_res[self._current_history_index][self._history_start_id + self._history_len]

        # how do we compare against the optimal congestion if we assume we know the future
        congestion_ratio = cost / optimal_congestion
        env_data[ExtraData.REWARD_OVER_FUTURE] = congestion_ratio

        self._history_start_id += 1
        observation = self._get_observation()

        logger.debug("cost  Congestion :{}".format(cost))
        logger.debug("optimal  Congestion :{}".format(optimal_congestion))
        logger.debug("Congestion Ratio :{}".format(congestion_ratio))

        if not congestion_ratio >= 1.0:
            assert error_bound(cost, optimal_congestion, 5e-4)
            logger.info("BUG!! congestion_ratio {}".format(congestion_ratio))
            congestion_ratio = 1.0

        reward = congestion_ratio * norm_factor

        done = self._is_terminal
        info = env_data
        self.diagnostics.append(info)
        return observation, reward, done, info

    def reset(self):
        self._history_start_id = 0

        self._current_history_index = (self._current_history_index + 1) % self._num_sequences
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def _get_reward(self, links_weights):
        tm = self._observations[self._current_history_index][self._history_start_id + self._history_len]
        cost, congestion_dict = self._optimizer.step(links_weights, tm)
        return cost
