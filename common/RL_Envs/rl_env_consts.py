class EnvsStrings:
    RL_ENV_SMART_NODES_GYM_ID = 'rl-smart_nodes-v0'


class EnvConsts:
    SOFTMIN_GAMMA = -2.0
    EPSILON = 1.0e-8
    INFTY = 1.0e8
    WEIGHT_LB = 1.0e-3
    WEIGHT_UB = 3.0e1
    WEIGHT_FACTOR = 1.0
    ZERO = 0.0
    GAMMA = 0.0


class HyperparamertsStrings:
    SOFTMIN_GAMMA = 'softmin_gamma'
    WEIGHT_LB = 'weight_lb'
    WEIGHT_UB = 'weight_ub'
    WEIGHT_FACTOR = 'weight_factor'

    LEARNING_RATE = 'learning_rate'
    BATCH_SIZE = 'batch_size'
    N_STEPS = 'n_steps'


class ExtraData:
    LOAD_PER_LINK = "load_per_link"
    CONGESTION_PER_LINK = "load_per_link"
    LINK_WEIGHTS = "links_weights"
    REWARD_OVER_FUTURE = "cost_over_future"
    MOST_CONGESTED_LINK = "most_congested_link"
