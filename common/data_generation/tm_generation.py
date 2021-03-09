from common.consts import TMType
import numpy as np
from functools import partial
from random import shuffle


def __gravity_generation(g, pairs, scale=1.0):
    all_gravity_flows = g.gravity_traffic_map(scale)
    return [flow for flow in all_gravity_flows if flow[0:2] in pairs]


def __uniform_generation(g, pairs, scale=1.0):
    all_gravity_flows = g.gravity_traffic_map(scale)
    lower_bound = min([flow for _, _, flow in all_gravity_flows])
    upper_bound = max([flow for _, _, flow in all_gravity_flows])
    return [(src, dst, scale * np.random.uniform(lower_bound, upper_bound)) for src, dst in pairs]


def __bimodal_generation(_, pairs, percent, big=400, small=150, std=20):
    flows = []

    shuffle(pairs)
    num_big_pairs_selected = int(np.ceil(len(pairs) * percent))

    for i, pair in enumerate(pairs):
        f_size_mb = -1
        while f_size_mb < 0:
            if i < num_big_pairs_selected:
                f_size_mb = np.random.normal(big, std)
            else:
                f_size_mb = np.random.normal(small, std)

        flows.append((pair[0], pair[1], f_size_mb))

    return flows


def __const_generation(_, pairs, const_value):
    flows = [(pair[0], pair[1], const_value) for pair in pairs]
    return flows


def __generate_tm(graph, matrix_sparsity, flow_generation_type, static_pairs=False, elephant_percentage=0.2, big=400,
                  small=150):
    if flow_generation_type == TMType.CONST:
        const_value = np.mean(graph.get_edges_capacities())
        get_flows = partial(__const_generation, const_value=const_value)
    elif flow_generation_type == TMType.BIMODAL:
        get_flows = partial(__bimodal_generation, percent=elephant_percentage, big=big, small=small)
    elif flow_generation_type == TMType.GRAVITY:
        get_flows = __gravity_generation
    elif flow_generation_type == TMType.UNIFORM:
        get_flows = __uniform_generation
    else:
        raise Exception("No exists traffic matrix type")

    pairs = graph.choosing_pairs(matrix_sparsity, static_pairs)
    return get_flows(graph, pairs)


def __raw_sample_mat(graph, matrix_sparsity, flow_generation_type, static_pairs=False, elephant_percentage=None,
                     big=400, small=150):
    tm = __generate_tm(graph, matrix_sparsity, flow_generation_type, static_pairs, elephant_percentage, big, small)
    num_nodes = graph.get_num_nodes

    tm_mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for src, dst, demand in tm:
        tm_mat[int(src), int(dst)] = max(0., demand)
    return tm_mat


def one_sample_tm_base(graph, matrix_sparsity, tm_type, static_pairs=False, elephant_percentage=0.2,
                       network_elephant=400,
                       network_mice=150):
    tm = __raw_sample_mat(graph, matrix_sparsity, tm_type, static_pairs, elephant_percentage, big=network_elephant,
                          small=network_mice)
    assert np.all(tm >= 0)
    return tm
