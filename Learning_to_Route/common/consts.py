'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class Consts:
    ACTION_OBLIVIOUS_PATH = 1
    ACTION_FLOW_PATH = 2

    INCREASE_ACTION = 0
    DECREASE_ACTION = 1
    NAK_ACTION = 2

    SRC_META_POS = 0
    DST_META_POS = 1
    EDGE_META_POS = 2
    WEIGHT_STR = 'weight'
    CAPACITY_STR = 'capacity'
    TTL_FLOW_STR = 'ttl_flow'

    # initial weight assignment strategy
    ONE_OVER_CAPACITY = 0
    RANDOM_WEIGHT = 1
    UNIT_WEIGHT = 2
    RANDOM_INT_WEIGHT = 3
    RANDOM_CYCLIC_INT_WEIGHT = 4
    LOCAL_SEARCH_WEIGHT = 5
    NOISY_LOCAL_SEARCH_WEIGHT = 6
    RANDOM_CYCLIC_NOISY_LOCAL_WEIGHT = 7
    MIXED_CYCLIC_NOISY_LOCAL_WEIGHT = 8
    MIXED_CYCLIC_RANDOM_INT_WEIGHT = 9

    # type of reward functions
    REWARD_EXP = 0
    REWARD_THUROP = 1
    REWARD_MAX_UTIL = 2

    THURP = 0
    NORMALIZED_THURP = 1

    MAX_UTIL = 2
    NORMALIZED_MAX_UTIL = 3

    DQN1 = "dqn1"
    DQN2 = "dqn2"
    PG = "pg"
    DDPG = "ddpg"

    DEATH_REWARD = -10 ** 4
    WIN_REWARD = 10 ** 4

    SAMPLE_ACTION = 1

    PG_EPOCH = 23
    MAX_WEIGHT = 50
    MIN_WEIGHT = 1

    TM_AMNT = 15

    Q_VALUE = 5


class EdgeConsts:
    WEIGHT_STR = 'weight'
    CAPACITY_STR = 'capacity'
    TTL_FLOW_STR = 'ttl_flow'

    MAX_WEIGHT = 50
    MIN_WEIGHT = 1

class TMType:
    # TMs types
    BIMODAL = "bimodal"
    GRAVITY = "gravity"