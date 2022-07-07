from common.utils import IS_LINUX

CALLBACK_PERFIX_PATH = '/home/idoye/PycharmProjects/Research_Implementing/Smart_Nodes_Routing/experiments/' if IS_LINUX \
    else 'C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Smart_Nodes_Routing\\experiments\\'

CALLBACK_PATH = lambda title: CALLBACK_PERFIX_PATH + '{}_callbacks_batch/'.format(title)


class EnvsStrings:
    RL_ENV_SMART_NODES_GYM_ID = 'rl-smart_nodes-v0'


class EnvConsts:
    SOFTMIN_GAMMA = -2.0
    WEIGHTS_FACTOR = 1.0
    EPSILON = 1.0e-8
    INFTY = 1.0e8
    WEIGHT_LB = 1.0e-3
    WEIGHT_UB = 3.0e1
    ZERO = 0.0
    GAMMA = 0


class HyperparamertsStrings:
    SOFTMIN_GAMMA = 'softMin_gamma'
    WEIGHTS_FACTOR = 'weights_factor'
    WEIGHT_LB = 'weight_lb'
    WEIGHT_UB = 'weight_ub'

    LEARNING_RATE = 'learning_rate'
    BATCH_SIZE = 'batch_size'
    N_STEPS = 'n_steps'


class ExtraData:
    LOAD_PER_LINK = "load_per_link"
    CONGESTION_PER_LINK = "load_per_link"
    LINK_WEIGHTS = "links_weights"
    REWARD_OVER_FUTURE = "cost_over_future"
    MOST_CONGESTED_LINK = "most_congested_link"
