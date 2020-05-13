from Learning_to_Route.common.consts import Consts
from Learning_to_Route.common.utils import norm_func
from random import shuffle
import numpy as np
from functools import partial


def __gravity_generation(g, pairs, scale=1.0):
    flows = []
    nodes = set()
    for p in pairs:
        nodes.add(p[0])
        nodes.add(p[1])

    capacity_map = {}
    ttl_capacity = 0
    for n in nodes:
        n_cap = 0
        neighs = g[n].keys()
        for neigh in neighs:
            n_cap += g[n][neigh][Consts.CAPACITY_STR]
        capacity_map[n] = n_cap
        ttl_capacity += n_cap

    for pair in pairs:
        src, dst = pair
        flows.append((src, dst,
                      scale * (capacity_map[src] * capacity_map[dst] / ttl_capacity)))

    return flows


def __bimodal_generation(g, pairs, percent, big=600, small=150, std=20):
    flows = []

    shuffle(pairs)
    num_big_pairs_selected = int(np.ceil(len(pairs) * percent))

    for i, pair in enumerate(pairs):
        fsize = -1
        while fsize < 0:
            if i < num_big_pairs_selected:
                fsize = np.random.normal(big, std)
            else:
                fsize = np.random.normal(small, std)

        flows.append((pair[0], pair[1], fsize))

    return flows


def __generate_tm(graph, matrix_sparsity, flow_generation_type, elephant_percentage=0.2, big=400, small=150):
    if flow_generation_type == Consts.BIMODAL:
        get_flows = partial(__bimodal_generation, percent=elephant_percentage, big=big, small=small)
    elif flow_generation_type == Consts.GRAVITY:
        get_flows = __gravity_generation
    else:
        raise Exception("No exists flow generation type")

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
    for f in tm:
        tm_mat[int(f[0]), int(f[1])] = max(0, f[2])
    return tm_mat


def one_sample_tm_base(graph, matrix_sparsity, tm_type, elephant_percentage, network_elephant, network_mice):
    tm = __raw_sample_mat(graph, matrix_sparsity, tm_type, elephant_percentage, big=network_elephant, small=network_mice)
    assert np.all(tm >= 0)
    return tm
