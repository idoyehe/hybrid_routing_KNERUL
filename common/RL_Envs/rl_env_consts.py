from common.utils import SEPERATOR
import os

CALLBACK_PERFIX_PATH = os.path.join(os.getcwd(),"experiments")

CALLBACK_PATH = lambda title: os.path.join(os.getcwd(),f"{title}_callbacks_batch{SEPERATOR}")


class EnvsStrings:
    RL_ENV_SMART_NODES_GYM_ID = 'rl-smart_nodes-v0'


class EnvConsts:
    SOFTMIN_GAMMA = -2.0
    WEIGHTS_FACTOR = 1.0
    EPSILON = 1.0e-8
    INFTY = 1.0e8
    WEIGHT_LB = 1
    WEIGHT_UB = 5
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
