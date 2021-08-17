'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class EnvConsts:
    SOFTMIN_ALPHA = -2.0
    EPSILON = 1.0e-8
    INFTY = 1.0e8
    WEIGHT_LB = 1.0e-3
    WEIGHT_UB = 3.0e1
    ZERO = 0.0

    ACTION_SPLITTINT_RATIOS = "splitting"
    ACTION_TM = "tm"

    DYNAMIC_LINK = "dynamic"
    STD_LINK_VALUE = "std"
    STD_MEAN_VALUE = "mean"


class ExtraData:
    LOAD_PER_LINK = "load_per_link"
    LINK_WEIGHTS = "links_weights"
    REWARD_OVER_FUTURE = "cost_over_future"
    MOST_CONGESTED_LINK = "most_congested_link"
    VS_OBLIVIOUS_DATA = "vs_oblivious_data"
