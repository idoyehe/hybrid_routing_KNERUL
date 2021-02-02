from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from consts import Consts
from utils import norm_func
from size_consts import SizeConsts

from functools import reduce
import numpy as np


def sample_tm_cyclic(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant,
                     network_mice, num_tms, **kwargs):
    q_val = kwargs['q'] if 'q' in kwargs else Consts.Q_VALUE
    res = []
    print("we are using q val:", q_val)
    for _ in range(q_val):
        res.append(one_sample_tm_base(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant, network_mice))

    while len(res) < num_tms:
        res.append(res[-q_val])

    return res


def sample_tm_average(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant, network_mice, num_tms, **kwargs):
    q_val = kwargs['q'] if 'q' in kwargs else Consts.Q_VALUE
    res = []
    print("we are using q val:", q_val)
    for _ in range(q_val):
        res.append(one_sample_tm_base(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant, network_mice))

    while len(res) < num_tms:
        res.append(reduce(lambda x, y: x + y, res[len(res) - q_val:]) / q_val)

    return res


def sample_tm_reg(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant, network_mice, num_tms, **kwargs):
    res = []
    for _ in range(num_tms):
        res.append(one_sample_tm_base(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant, network_mice))

    return res


def get_x_y_data(graph, matrix_sparsity, num_histories, get_xy,
                 tm_type=Consts.GRAVITY, max_history_len=50,
                 history_window=10, elephant_perc=0.2, mice_sz=1e-4, elephant_sz=1.5e-2,
                 **kwargs):
    X = []
    Y = []
    avg_node_capacity = norm_func(graph.get_avg_g_cap(), 1.0 * SizeConsts.ONE_Mb)
    network_elephant = elephant_sz * avg_node_capacity
    network_mice = mice_sz * avg_node_capacity

    for _ in range(num_histories):

        res_norm = get_xy(graph, matrix_sparsity, tm_type, elephant_perc, network_elephant, network_mice,
                          max_history_len + history_window + 1, **kwargs)

        x = []
        y = []
        for start_ind in range(len(res_norm) - history_window):  # was without -history_window
            hist = res_norm[start_ind:start_ind + history_window]
            next_tm = res_norm[start_ind + history_window]
            x.append(np.stack(hist, 0))
            y.append(next_tm.flatten())

        X += x
        Y += y
    return np.asarray(X), np.asarray(Y)

