from Learning_to_Route.common.size_consts import SizeConsts
from numpy import round, where, dstack, abs


def norm_func(x, norm_val=1. * SizeConsts.ONE_Mb):
    return x / norm_val


def to_int(f: float):
    return int(round(f))


def extract_flows(traffic_matrix):
    return list(map(lambda f: tuple(f), dstack(where(traffic_matrix > 0))[0]))


def error_bound(x, y=0, ERROR_BOUND=1e-10):
    return abs(x - y) <= ERROR_BOUND
