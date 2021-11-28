from common.consts import TMType
from common.network_class import NetworkClass
import numpy as np
from functools import partial
from random import shuffle


def __gravity_generation(g, pairs, scale=1.0):
    all_gravity_flows = g.gravity_traffic_map(scale)
    return [flow for flow in all_gravity_flows if flow[0:2] in pairs]


def __uniform_generation(net: NetworkClass, pairs, scale=1.0):
    all_gravity_flows = net.gravity_traffic_map(scale)
    lower_bound = min([flow for _, _, flow in all_gravity_flows])
    upper_bound = max([flow for _, _, flow in all_gravity_flows])
    return [(src, dst, scale * np.random.uniform(lower_bound, upper_bound)) for src, dst in pairs]


def __poisson_generation(net: NetworkClass, pairs, scale=1.0):
    all_gravity_flows = net.gravity_traffic_map(scale)
    lower_bound = min([flow for _, _, flow in all_gravity_flows])
    upper_bound = max([flow for _, _, flow in all_gravity_flows])
    flows_list = list()
    for src, dst in pairs:
        _poisson_lambda = np.random.uniform(lower_bound, upper_bound)
        flows_list.append((src, dst, np.random.poisson(_poisson_lambda)))
    return flows_list


def __bimodal_generation(net: NetworkClass, pairs, g_1_ratio, g_1=(800, 100), g_2=(400, 100)):
    flows = []
    elephant_percentages = net.elephant_percentages()
    g_1_mean, g_1_std = g_1
    g_2_mean, g_2_std = g_2
    shuffle(pairs)
    num_g_1_pairs_selected = int(np.ceil(len(pairs) * g_1_ratio))
    for i, (src, dst) in enumerate(pairs):
        coin = np.random.random_sample()
        ep_flag = coin <= elephant_percentages[src]
        f_size = -1
        while f_size < 0:
            if ep_flag:
                f_size = np.random.normal(g_1_mean, g_1_std)
            else:
                f_size = np.random.normal(g_2_mean, g_2_std)

        flows.append((src, dst, f_size))

    return flows


def __const_generation(_, pairs, const_value):
    flows = [(pair[0], pair[1], const_value) for pair in pairs]
    return flows


def __generate_tm(graph, matrix_sparsity, flow_generation_type, static_pairs, g_1_ratio, g_1, g_2):
    if flow_generation_type == TMType.CONST:
        const_value = np.mean(graph.get_edges_capacities()) / 10
        get_flows = partial(__const_generation, const_value=const_value)
    elif flow_generation_type == TMType.BIMODAL:
        get_flows = partial(__bimodal_generation, g_1_ratio=g_1_ratio, g_1=g_1, g_2=g_2)
    elif flow_generation_type == TMType.GRAVITY:
        get_flows = __gravity_generation
    elif flow_generation_type == TMType.UNIFORM:
        get_flows = __uniform_generation
    elif flow_generation_type == TMType.POISSON:
        get_flows = __poisson_generation
    else:
        raise Exception("No exists traffic matrix type")

    pairs = graph.choosing_pairs(matrix_sparsity, static_pairs)
    return get_flows(graph, pairs)


def __raw_sample_mat(graph, matrix_sparsity, flow_generation_type, static_pairs, g_1_ratio, g_1, g_2):
    tm = __generate_tm(graph, matrix_sparsity, flow_generation_type, static_pairs, g_1_ratio, g_1, g_2)
    num_nodes = graph.get_num_nodes

    tm_mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for src, dst, demand in tm:
        tm_mat[int(src), int(dst)] = max(0., demand)
    return tm_mat


def one_sample_tm_base(graph, matrix_sparsity, tm_type, static_pairs=False, g_1_ratio=0.2, g_1=(800, 100), g_2=(400, 100)):
    tm = __raw_sample_mat(graph, matrix_sparsity, tm_type, static_pairs, g_1_ratio, g_1, g_2)
    assert np.all(tm >= 0)
    return tm
