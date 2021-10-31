'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class Consts:
    ZERO = 1e-2
    ERROR_BOUND = 1e-3
    OUTPUT_FLAG = 1
    FEASIBILITY_TOL = 1e-7
    PRIMAL_SIMPLEX = 0
    DUAL_SIMPLEX = 1
    BARRIER_METHOD = 2
    NUMERIC_FOCUS = 3
    CROSSOVER = -1
    SCALE = 1e1
    BAR_CONV_TOL = 1e-4
    MAX_ITER = 500
    ROUND = 6
    WEIGHT_STR = 'weight'
    CAPACITY_STR = 'capacity'

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
    CONST = "const"
    BIMODAL = "bimodal"
    GRAVITY = "gravity"
    UNIFORM = "uniform"
