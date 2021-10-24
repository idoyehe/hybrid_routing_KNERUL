from common.size_consts import SizeConsts
from common.consts import Consts
from numpy import round, where, dstack, abs
from pathlib import Path
import pickle
import numpy as np
import itertools
from platform import system
import torch
IS_LINUX = system() == "Linux"

SEPERATOR = "/" if IS_LINUX else "\\"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_lp_values(gurobi_vars_dict, R=Consts.ROUND):
    gurobi_vars_dict = dict(gurobi_vars_dict)

    for key in gurobi_vars_dict.keys():
        gurobi_vars_dict[key] = np.round(gurobi_vars_dict[key].x, R)
    return gurobi_vars_dict


def norm_func(x, norm_val=1. * SizeConsts.ONE_Mb):
    return x / norm_val


def to_int(f: float):
    return int(round(f))


def extract_flows(traffic_matrix):
    return list(map(lambda f: tuple(f), dstack(where(traffic_matrix > 0.0))[0]))


def error_bound(x, y=0, error_bound=Consts.ERROR_BOUND):
    return abs(x - y) <= error_bound


def load_dump_file(file_name: str):
    dumped_file = open(Path(file_name), 'rb')
    dict2load = pickle.load(dumped_file)
    dumped_file.close()
    assert isinstance(dict2load, dict)
    return dict2load


def change_zero_cells(tm):
    assert tm.shape[0] == tm.shape[1]
    tm[tm == 0] = Consts.ZERO
    np.fill_diagonal(tm, 0)
    return tm


def find_nodes_subsets(_set, size):
    return list(itertools.combinations(_set, size))
