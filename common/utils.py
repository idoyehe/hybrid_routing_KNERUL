from size_consts import SizeConsts
from numpy import round, where, dstack, abs
from pathlib import Path
import pickle


def norm_func(x, norm_val=1. * SizeConsts.ONE_Mb):
    return x / norm_val


def to_int(f: float):
    return int(round(f))


def extract_flows(traffic_matrix):
    return list(map(lambda f: tuple(f), dstack(where(traffic_matrix > 0))[0]))


def error_bound(x, y=0, ERROR_BOUND=1e-4):
    return abs(x - y) <= ERROR_BOUND


def load_dump_file(file_name: str):
    dumped_file = open(Path(file_name), 'rb')
    dict2load = pickle.load(dumped_file)
    dumped_file.close()
    assert isinstance(dict2load, dict)
    return dict2load
