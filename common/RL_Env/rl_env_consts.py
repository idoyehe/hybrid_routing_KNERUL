'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class HistoryConsts:
    SOFTMIN_ALPHA = -2.0
    SOFTMAX_ALPHA = 1.0
    EPSILON = 1.0e-10
    PERC_DEMAND = 0.99999999999999999999999999999999
    INFTY = 1.0e4
    ZERO = 0.0

    ACTION_SPLITTINT_RATIOS = "splitting"
    ACTION_W_EPSILON = "w_eps"
    ACTION_W_INFTY = "w_inf"
    ACTION_TM = "tm"

    ACTIONS_W = [ACTION_W_EPSILON, ACTION_W_INFTY]

    DYNAMIC_LINK = "dynamic"
    STD_LINK_VALUE = "std"
    STD_MEAN_VALUE = "mean"


class ExtraData:
    LOAD_PER_LINK = "load_per_link"
    LINK_WEIGHTS = "links_weights"
    REWARD_OVER_FUTURE = "cost_over_future"
    MOST_CONGESTED_LINK = "most_congested_link"
    VS_OBLIVIOUS_DATA = "vs_oblivious_data"