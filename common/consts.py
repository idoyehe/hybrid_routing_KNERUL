'''
Created on 6 Feb 2017

@author: asafvaladarsky
'''


class Consts:
    ZERO = 1e-2
    ERROR_BOUND = 1e-3
    OUTPUT_FLAG = 0
    FEASIBILITY_TOL = 1e-8
    PRIMAL_SIMPLEX = 0
    DUAL_SIMPLEX = 1
    BARRIER_METHOD = 2
    NUMERIC_FOCUS = 3
    CROSSOVER = -1
    SCALE = 1e1
    BAR_CONV_TOL = 1e-4
    MAX_ITER = 500
    ROUND = 6
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


class DumpsConsts:
    TMs = "tms"
    NET_PATH = "net_path"
    EXPECTED_CONGESTION = "expected_congestion"
    INITIAL_WEIGHTS = "initial_weights"
    OPTIMAL_SPLITTING_RATIOS = "optimal_splitting_ratios"
    DEST_EXPECTED_CONGESTION = "dest_expected_congestion"
    MATRIX_SPARSITY = "matrix_sparsity"
    MATRIX_TYPE = "matrix_type"
