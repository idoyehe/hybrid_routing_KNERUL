from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from Learning_to_Route.common.consts import Consts
from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass, EdgeConsts
from topologies import topologies, topology_zoo_loader
from docplex.mp.model import Model


def oblivious_routing(net: NetworkClass):
    m = Model(name="Applegate's and Cohen's LP formulations")
    reduced_directed = net.get_graph.to_directed()
    r = m.continuous_var(name="r")
    m.minimize(r)

    pi_edges_dict = defaultdict(dict)
    pe_edges_dict = defaultdict(dict)
    f_arch_dict = defaultdict(dict)

    out_archs = defaultdict(list)
    in_archs = defaultdict(list)
    for _e in net.get_graph.edges:
        for _h in net.get_graph.edges:
            pi_edges_dict[_e][_h] = m.continuous_var(name="PI_{}_{}".format(_e, _h), lb=0)

    for _e in net.get_graph.edges:
        for i in range(net.get_num_nodes):
            for j in range(net.get_num_nodes):
                if i == j:
                    pe_edges_dict[_e][(i, j)] = 0
                else:
                    pe_edges_dict[_e][(i, j)] = m.continuous_var(name="PE_{}_{}".format(_e, (i, j)), lb=0)

    for _arc in reduced_directed.edges:
        for i in range(net.get_num_nodes):
            for j in range(net.get_num_nodes):
                if i == j:
                    f_arch_dict[_arc][(i, j)] = 0
                else:
                    f_arch_dict[_arc][(i, j)] = m.continuous_var(name="f_{}_{}".format(_arc, (i, j)), lb=0)

        in_archs[_arc[1]].append(_arc)
        out_archs[_arc[0]].append(_arc)

    for i in range(net.get_num_nodes):
        for j in range(net.get_num_nodes):
            if i == j:
                continue

            f_out_arch_i = [f_arch_dict[_out_arch][(i, j)] for _out_arch in out_archs[i]]
            f_in_arch_i = [f_arch_dict[_in_arch][(i, j)] for _in_arch in in_archs[i]]

            m.add_constraint(m.sum(f_out_arch_i) - m.sum(f_in_arch_i) == 1)

    for k in range(net.get_num_nodes):
        for i in range(net.get_num_nodes):
            if i == k:
                continue
            for j in range(net.get_num_nodes):
                if i == j or j == k:
                    continue

                f_out_arch_k = [f_arch_dict[_out_arch][(i, j)] for _out_arch in out_archs[k]]
                f_in_arch_k = [f_arch_dict[_in_arch][(i, j)] for _in_arch in in_archs[k]]

                m.add_constraint(m.sum(f_out_arch_k) - m.sum(f_in_arch_k) == 0)

    for _e in net.get_graph.edges:
        _e_list_sum = []
        for _h in net.get_graph.edges:
            cap_h = net.get_edge_key(_h, EdgeConsts.CAPACITY_STR)
            _e_list_sum.append(cap_h * pi_edges_dict[_e][_h])
        m.add_constraint(m.sum(_e_list_sum) <= r)

    for _e in net.get_graph.edges:
        for i in range(net.get_num_nodes):
            for _arc in reduced_directed.edges:
                _edge_e = _arc
                j = _arc[0]
                k = _arc[1]
                if _arc[0] > _arc[1]:
                    _edge_e = (_arc[1], _arc[0])

                m.add_constraint((pi_edges_dict[_e][_edge_e] + pe_edges_dict[_e][(i, j)] - pe_edges_dict[_e][(i, k)]) >= 0)

    for _e in net.get_graph.edges:
        _cap_e = net.get_edge_key(_e, EdgeConsts.CAPACITY_STR)
        for i in range(net.get_num_nodes):
            for j in range(net.get_num_nodes):
                if i == j:
                    continue
                f_e = f_arch_dict[_e][(i, j)] + f_arch_dict[(_e[1], _e[0])][(i, j)]
                m.add_constraint(f_e / _cap_e <= pe_edges_dict[_e][(i, j)])

    m.solve()
    m.print_solution()


ecmpNetwork = NetworkClass(topologies["TRIANGLE"])
# ecmpNetwork = ECMPNetwork(topology_zoo_loader("http://www.topology-zoo.org/files/Ibm.gml", default_capacity=45))
routing = {
    (0, 1): [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
    (0, 2): [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
    (1, 2): [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]}

oblivious_routing(ecmpNetwork)
pass
