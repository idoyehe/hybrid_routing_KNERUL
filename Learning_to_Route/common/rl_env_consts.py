'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class HistoryConsts:
    SOFTMIN_ALPHA = -2.0
    SOFTMAX_ALPHA = 1.0
    EPSILON = 1.0e-2
    PERC_DEMAND = 0.999
    INFTY = 1.0e4
    ZERO = 0.0

    ACTION_SPLITTINT_RATIOS = "splitting"
    ACTION_W_EPSILON = "w_eps"
    ACTION_W_INFTY = "w_inf"
    ACTION_TM = "tm"

    ACTIONS_W = [ACTION_W_EPSILON, ACTION_W_INFTY]


class ExtraData:
    REWARD_OVER_FUTURE = "over_future"
    REWARD_OVER_PREV = "over_prev"
    REWARD_OVER_AVG = "over_avg"
    REWARD_OVER_AVG_EXPECTED = "over_avg_expected"
    REWARD_OVER_AVG_ACTUAL = "over_avg_actual"
    REWARD_OVER_RANDOM = "over_random"


class WeightStrategy:
    # initial weight assignment strategy
    ONE_OVER_CAPACITY = "one_over"
    RANDOM_WEIGHT = "random"
    UNIT_WEIGHT = "unit"
    RANDOM_INT_WEIGHT = "random_int"
    RANDOM_CYCLIC_INT_WEIGHT = "random_cyc_int"
    LOCAL_SEARCH_WEIGHT = "ls"
    NOISY_LOCAL_SEARCH_WEIGHT = "noisy_ls"
    RANDOM_CYCLIC_NOISY_LOCAL_WEIGHT = "random_cyc_noisy_ls"
    MIXED_CYCLIC_NOISY_LOCAL_WEIGHT = "static_random_ls"
    MIXED_CYCLIC_RANDOM_INT_WEIGHT = "static_random"
    MIXED_STATIC_THEN_LOCAL = "static_ls"


class GraphConsts:
    SRC_META_POS = 0
    DST_META_POS = 1
    EDGE_META_POS = 2


class RewardType:
    # type of reward functions
    REWARD_EXP = "exp"
    REWARD_THUROP = "thurop"
    REWARD_MAX_UTIL = "max_util"
    REWARD_AVG_UTIL = "avg_util"


class ActionType:
    ACTION_OBLIVIOUS_PATH = "oblivious"
    ACTION_FLOW_PATH = "with_flows"


class ActionIDs:
    INCREASE_ACTION = 0
    DECREASE_ACTION = 1
    NAK_ACTION = 2


class GeneralConsts:
    DEATH_REWARD = -10 ** 4
    WIN_REWARD = 10 ** 4

    SAMPLE_ACTION = 1

    PG_EPOCH = 1  # 23
