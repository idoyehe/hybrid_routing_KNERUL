'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class HistoryConsts:
    SOFTMIN_ALPHA = -2.0
    SOFTMAX_ALPHA = 1.0
    EPSILON = 1.0e-10
    PERC_DEMAND = 0.999999999999999999
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
    REWARD_OVER_FUTURE = "over_future"
    REWARD_OVER_PREV = "over_prev"
    REWARD_OVER_AVG = "over_avg"
    REWARD_OVER_AVG_EXPECTED = "over_avg_expected"
    REWARD_OVER_AVG_ACTUAL = "over_avg_actual"
    REWARD_OVER_RANDOM = "over_random"
