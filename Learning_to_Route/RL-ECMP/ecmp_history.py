"""
Created on 26 Jun 2017
@author: asafvaladarsky

refactoring on 26/04/2020
@by: Ido Yehezkel
"""
from Learning_to_Route.common.utils import error_bound
from gym import Env, spaces, envs, register
from common.network_class import *
from optimizer import WNumpyOptimizer
from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from common.rl_env_consts import HistoryConsts, ExtraData
from static_routing.generating_tms_dumps import load_dump_file
from static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver
from common.topologies import topology_zoo_loader
import random


class ECMPHistoryEnv(Env):

    def __init__(self,
                 max_steps,
                 ecmp_topo=None,
                 path_dumped=None,
                 history_length=None,
                 history_action_type=None,
                 train_histories_length=None,
                 test_histories_length=None,
                 tm_type=None,
                 tm_sparsity=None,
                 elephant_flows_percentage=None,
                 elephant_flow=None,
                 mice_flow=None,
                 testing=False):

        self._episode_len = max_steps

        if path_dumped is None:
            self._network = NetworkClass(ecmp_topo)
            self._tms = None
            self._tm_type = tm_type
            self._tm_sparsity_list = tm_sparsity  # percentage of participating pairs, assumed to be a list
        else:
            loaded_dict = load_dump_file(file_name=path_dumped)
            self._network = NetworkClass(
                topology_zoo_loader(url=loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
            self._tms = loaded_dict["tms"]
            self._tm_type = loaded_dict["tms_type"]
            self._tm_sparsity_list = [
                loaded_dict["tms_sparsity"]]  # percentage of participating pairs, assumed to be a list

        self._network = self._network.get_g_directed
        self._g_name = self._network.get_name
        self._num_edges = self._network.get_num_edges
        self._num_nodes = self._network.get_num_nodes
        self._all_pairs = self._network.get_all_pairs()
        self._history_start_id = 0
        self._current_history_index = -1
        self.current_sequence = 0
        self._optimizer = WNumpyOptimizer(self._network)

        self._history_len = history_length  # number of each state history
        self._history_action_type = history_action_type
        self._num_train_histories = train_histories_length  # number of different train seniors per sparsity
        self._num_test_histories = test_histories_length  # number of different test seniors per sparsity
        self._mice_flow = mice_flow
        self._elephant_flow = elephant_flow
        self._elephant_flows_percentage = elephant_flows_percentage

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
        self._init_random_baseline()
        self.diagnostics = []

    def get_num_steps(self):
        return self._episode_len

    def _set_observation_space(self):
        self._observation_space = spaces.Box(low=0.0, high=np.inf,
                                             shape=(self._history_len, self._num_nodes, self._num_nodes))

    def _set_action_space(self):
        self._action_space = spaces.Box(low=0, high=np.inf, shape=(self._num_edges,))

    def _sample_tm(self, p):
        # we need to make the TM change slowly in time, currently it changes every step kind of drastically

        if self._tms is None:
            tm = one_sample_tm_base(self._network, p, self._tm_type, self._elephant_flows_percentage,
                                    self._elephant_flow,
                                    self._mice_flow)
            opt, _ = optimal_load_balancing_LP_solver(self._network, tm)
        else:
            tm, opt = random.choice(self._tms)
        return tm, opt

    def _init_all_observations(self):
        self._train_observations = []
        self._test_observations = []
        self._opt_train_observations = []
        self._opt_test_observations = []

        for _ in range(self._num_train_histories):
            for p in self._tm_sparsity_list:
                train_episode = list()
                train_episode_optimal = list()
                for _ in range(self._history_len + self._episode_len):
                    tm, opt = self._sample_tm(p)
                    train_episode.append(tm)
                    train_episode_optimal.append(opt)
                self._train_observations.append(train_episode)
                self._opt_train_observations.append(train_episode_optimal)

        for _ in range(self._num_test_histories):
            test_episode = list()
            test_episode_optimal = list()
            for p in self._tm_sparsity_list:
                for _ in range(self._history_len + self._episode_len):
                    tm, opt = self._sample_tm(p)
                    test_episode.append(tm)
                    test_episode_optimal.append(opt)
                self._test_observations.append(test_episode)
                self._opt_test_observations.append(test_episode_optimal)

        self._actual_num_train_histories = len(self._train_observations)
        self._actual_num_test_histories = len(self._test_observations)

        # self._validate_data()

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
        self._random_res = self._random_test_res if testing else self._random_train_res

    def _process_action(self, action):
        if self._history_action_type == HistoryConsts.ACTION_W_EPSILON:
            action[action <= 0] = HistoryConsts.EPSILON
        elif self._history_action_type == HistoryConsts.ACTION_W_INFTY:
            action[action <= 0] = HistoryConsts.INFTY
        return action

    def _init_random_baseline(self):
        def populate(observations, res_dict, train_res, std_res):
            action_size = self._action_space.shape[0]
            for train_index, tm_list in enumerate(observations):
                res_dict[train_index] = {}
                res_avg_tm = []
                res_std_tm = []
                for tm_index, tm in enumerate(tm_list):
                    res_dict[train_index][tm_index] = []
                    for _ in range(5):
                        action = np.random.rand(action_size)
                        action = self._process_action(action)
                        cost, _ = self._optimizer.step(action, tm)
                        reward = -1 * cost
                        res_dict[train_index][tm_index].append(reward)

                    res_avg_tm.append(np.average(res_dict[train_index][tm_index]))
                    res_std_tm.append(np.std(res_dict[train_index][tm_index]))
                train_res.append(res_avg_tm)
                std_res.append(res_std_tm)

        populate(self._train_observations, self._random_train,
                 self._random_train_res, self._random_train_std)

        populate(self._test_observations, self._random_test,
                 self._random_test_res, self._random_test_std)

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
        self._w = self._process_action(action)

        cost = self._get_reward()
        self._is_terminal = self._history_start_id + 1 == self._episode_len

        norm_factor = -1

        env_data = {}
        optimal_congestion = self._opt_res[self._current_history_index][
            self._history_start_id + self._history_len]
        # how do we compare against the optimal congestion if we assume we know the future
        env_data[ExtraData.REWARD_OVER_FUTURE] = cost / optimal_congestion
        env_data[ExtraData.REWARD_OVER_PREV] = cost / self._opt_res[self._current_history_index][
            self._history_start_id - 1 + self._history_len]
        env_data[ExtraData.REWARD_OVER_RANDOM] = cost / self._random_res[self._current_history_index][
            self._history_start_id + self._history_len]

        self._history_start_id += 1
        observation = self._get_observation()
        congestion_ratio = env_data[ExtraData.REWARD_OVER_FUTURE]

        # print("cost  Congestion :{}".format(cost))
        # print("optimal  Congestion :{}".format(optimal_congestion))
        # print("Congestion Ratio :{}".format(congestion_ratio))

        if not congestion_ratio >= 1.0:
            assert error_bound(cost, optimal_congestion, 5e-4)
            congestion_ratio = 1.0

        reward = congestion_ratio * norm_factor

        done = self._is_terminal
        info = env_data
        self.diagnostics.append(info)
        return observation, reward, done, info

    def reset(self):
        self._history_start_id = 0

        if (self._current_history_index + 1) == self._num_sequences:
            self.current_sequence += 1

        self._current_history_index = (self._current_history_index + 1) % self._num_sequences
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def _get_reward(self):
        tm = self._observations[self._current_history_index][self._history_start_id + self._history_len]
        cost, congestion_dict = self._optimizer.step(self._w, tm)
        return cost


ECMP_ENV_GYM_ID: str = 'ecmp-history-v0'

if ECMP_ENV_GYM_ID not in envs.registry.env_specs:
    register(id=ECMP_ENV_GYM_ID,
             entry_point='ecmp_history:ECMPHistoryEnv',
             kwargs={
                 'max_steps': 1,
                 'history_length': 1,
                 'path_dumped': "/home/idoye/PycharmProjects/Research_Implementing/Learning_to_Route/TMs_DB/T-lex_tms_12X12_length_20000_gravity_sparsity_0.3",
                 'train_histories_length': 5000,
                 'test_histories_length': 0,
                 'history_action_type': HistoryConsts.ACTION_W_EPSILON}
             )
