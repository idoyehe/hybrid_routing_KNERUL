class Consts:
    ZERO = 1e-2
    ERROR_BOUND = 1e-3
    OUTPUT_FLAG = 1
    FEASIBILITY_TOL = 1e-6
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
    CUSTOM_BIMODAL = "custom_bimodal"
    BIMODAL = "bimodal"
    GRAVITY = "gravity"
    UNIFORM = "uniform"
    POISSON = "poisson"
    REAL = "real_traffic"


class DumpsConsts:
    TMs = "tms"
    NET_PATH = "url"
    EXPECTED_CONGESTION = "expected_congestion"
    INITIAL_WEIGHTS = "initial_weights"
    OPTIMAL_SPLITTING_RATIOS = "optimal_splitting_ratios"
    DEST_EXPECTED_CONGESTION = "dest_expected_congestion"
    MATRIX_SPARSITY = "matrix_sparsity"
    MATRIX_TYPE = "matrix_type"
    G_1_RATIO = "g_1_ratio"
    OBLIVIOUS_RATIO = "oblivious_ratio"
    OBLIVIOUS_MEAN_CONGESTION = "oblivious_mean_congestion"
    OBLIVIOUS_SRC_DST_SPR = "oblivious_src_dst_spr"
    COPE_RATIO = "cope_ratio"
    COPE_MEAN_CONGESTION = "cope_mean_congestion"
    COPE_SRC_DST_SPR = "cope_src_dst_spr"
