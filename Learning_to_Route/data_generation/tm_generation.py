from Learning_to_Route.common.consts import TMType, Consts
from Learning_to_Route.common.utils import *
from random import shuffle
import numpy as np
from functools import partial


def __gravity_generation(g, pairs, scale=1.0):
    flows = []
    included_nodes = set()
    for p in pairs:
        included_nodes.add(p[0])
        included_nodes.add(p[1])

    capacity_map = {}
    total_capacity: float = 0.0
    for node in included_nodes:
        node_out_cap = sum(out_edge[2][Consts.CAPACITY_STR] for out_edge in g.out_edges_by_node(node, data=True))
        capacity_map[node] = node_out_cap
        total_capacity += node_out_cap

    for pair in pairs:
        src, dst = pair
        f_size = to_int(capacity_map[src] * capacity_map[dst] / total_capacity)
        flows.append((src, dst, scale * f_size))

    return flows


def __bimodal_generation(graph, pairs, percent, big=400, small=150, std=20):
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


def __customize_generation(g, pairs, mean, std):
    flows = []

    for pair in pairs:
        f_size_mb = -1
        while f_size_mb < 0:
            f_size_mb = np.mean([1477.6729,121674.91])

        flows.append((pair[0], pair[1], f_size_mb))

    return flows

def __event_generation(graph, pairs, f_size_mb):
    flows = [(pair[0], pair[1], f_size_mb) for pair in pairs]
    return flows

def __const_generation(_, pairs, const_value):
    flows = [(pair[0], pair[1], const_value) for pair in pairs]
    return flows


def __generate_tm(graph, matrix_sparsity, flow_generation_type, elephant_percentage=0.2, big=400, small=150):
    if flow_generation_type == TMType.CONST:
        const_value = np.mean(graph.get_edges_capacities())
        matrix_sparsity = 1.0
        get_flows = partial(__const_generation, const_value=const_value)
    elif flow_generation_type == TMType.BIMODAL:
        get_flows = partial(__bimodal_generation, percent=elephant_percentage, big=big, small=small)
    elif flow_generation_type == TMType.GRAVITY:
        get_flows = __gravity_generation
    elif flow_generation_type == TMType.CUSTOMIZE:
        mean = np.mean(graph.get_edges_capacities())
        std = np.std(graph.get_edges_capacities()) * 10
        get_flows = partial(__customize_generation, mean=mean, std=std)
    elif flow_generation_type == TMType.RARE_EVENT:
        coin = np.random.uniform()
        get_flows = __gravity_generation
        if coin <= 0.2:
            value = np.max(graph.get_edges_capacities())
            get_flows = partial(__event_generation, f_size_mb=value)
    else:
        raise Exception("No exists traffic matrix type")

    all_pairs = list(graph.get_all_pairs())

    # shuffle the pairs
    shuffle(all_pairs)
    num_pairs_selected = int(np.ceil(len(all_pairs) * matrix_sparsity))
    pairs = []
    while len(pairs) != num_pairs_selected:
        new_ind = np.random.choice(len(all_pairs))
        pairs.append(all_pairs[new_ind])
        all_pairs.pop(new_ind)
    return get_flows(graph, pairs)


def __raw_sample_mat(graph, matrix_sparsity, flow_generation_type, elephant_percentage=None, big=400, small=1):
    tm = __generate_tm(graph, matrix_sparsity, flow_generation_type, elephant_percentage, big, small)
    num_nodes = graph.get_num_nodes

    tm_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst, demand in tm:
        tm_mat[int(src), int(dst)] = max(0., demand)
    return tm_mat


def one_sample_tm_base(graph, matrix_sparsity, tm_type, elephant_percentage=0.2, network_elephant=400,
                       network_mice=150):
    tm = __raw_sample_mat(graph, matrix_sparsity, tm_type, elephant_percentage, big=network_elephant,
                          small=network_mice)
    assert np.all(tm >= 0)
    return tm
