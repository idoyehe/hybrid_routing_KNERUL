class Consts:
    ZERO = 1e-2
    ERROR_BOUND = 1e-3
    OUTPUT_FLAG = 1
    FEASIBILITY_TOL = 1e-7
    PRIMAL_SIMPLEX = 0
    DUAL_SIMPLEX = 1
    PRIMAL_DUAL_SIMPLEX = 5
    PRESOLVE = 2
    BARRIER_METHOD = 2
    NUMERIC_FOCUS = 3
    CROSSOVER = 0
    SCALE_FLAG = 3
    SCALE = 1e2
    BAR_CONV_TOL = 1e-4
    MAX_ITER = 500
    ROUND = 6
    Q_VALUE = 5


class EdgeConsts:
    WEIGHT_STR = 'weight'
    CAPACITY_STR = 'capacity'


class TMType:
    # TMs types
    CONST = "const"
    BIMODAL = "bimodal"
    GRAVITY = "gravity"
    UNIFORM = "uniform"
    POISSON = "poisson"


class DumpsConsts:
    TMs = "tms"
    NET_PATH = "url"
    EXPECTED_CONGESTION = "expected_congestion"
    INITIAL_WEIGHTS = "initial_weights"
    OPTIMAL_SPLITTING_RATIOS = "optimal_splitting_ratios"
    DEST_EXPECTED_CONGESTION = "dest_expected_congestion"
    MATRIX_SPARSITY = "matrix_sparsity"
    MATRIX_TYPE = "matrix_type"
    OBLIVIOUS_RATIO = "oblivious_ratio"
    OBLIVIOUS_MEAN_CONGESTION = "oblivious_mean_congestion"
    OBLIVIOUS_SRC_DST_SPR = "oblivious_src_dst_spr"
