from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass, EdgeConsts
from topologies import topologies, topology_zoo_loader
from docplex.mp.model import Model


def oblivious_routing(net: NetworkClass):
    m = Model(name="Applegate's and Cohen's Oblivious Routing LP formulations")
    reduced_directed = net.get_graph.to_directed()
    r = m.continuous_var(name="r")
    m.minimize(r)

    pi_edges_dict = defaultdict(dict)
    pe_edges_dict = defaultdict(dict)
    f_arch_dict = defaultdict(dict)

    out_arches = defaultdict(list)
    in_arches = defaultdict(list)
    for _e in net.edges:
        _e_list_sum = []
        for _h in net.edges:
            pi_edges_dict[_e][_h] = m.continuous_var(name="PI_{}_{}".format(_e, _h), lb=0)
            cap_h = net.get_edge_key(_h, EdgeConsts.CAPACITY_STR)
            _e_list_sum.append(cap_h * pi_edges_dict[_e][_h])
        m.add_constraint(m.sum(_e_list_sum) <= r)

    for _e in net.edges:
        _capacity_e = net.get_edge_key(_e, EdgeConsts.CAPACITY_STR)
        _arch = _e
        _reversed_arch = (_e[1], _e[0])
        for i in range(net.get_num_nodes):
            for j in range(net.get_num_nodes):
                if i == j:
                    pe_edges_dict[_e][(i, j)] = 0
                    f_arch_dict[_arch][(i, j)] = 0
                    f_arch_dict[_reversed_arch][(i, j)] = 0
                else:
                    pe_edges_dict[_e][(i, j)] = m.continuous_var(name="PE_{}_{}".format(_e, (i, j)), lb=0)
                    f_arch_dict[_arch][(i, j)] = m.continuous_var(name="f_{}_{}".format(_arch, (i, j)), lb=0)
                    f_arch_dict[_reversed_arch][(i, j)] = m.continuous_var(name="f_{}_{}".format(_reversed_arch, (i, j)), lb=0)

                    f_e = f_arch_dict[_arch][(i, j)] + f_arch_dict[_reversed_arch][(i, j)]
                    m.add_constraint(f_e / _capacity_e <= pe_edges_dict[_e][(i, j)])

        in_arches[_arch[1]].append(_arch)
        out_arches[_arch[0]].append(_arch)

        in_arches[_reversed_arch[1]].append(_reversed_arch)
        out_arches[_reversed_arch[0]].append(_reversed_arch)

    # flow constrains
    for k in net.nodes:
        for i in net.nodes:
            for j in net.nodes:
                if i == j or j == k:
                    continue

                f_out_arch_k = [f_arch_dict[_out_arch][(i, j)] for _out_arch in out_arches[k]]
                f_in_arch_k = [f_arch_dict[_in_arch][(i, j)] for _in_arch in in_arches[k]]
                if i == k:
                    m.add_constraint(m.sum(f_out_arch_k) - m.sum(f_in_arch_k) == 1)
                else:
                    m.add_constraint(m.sum(f_out_arch_k) - m.sum(f_in_arch_k) == 0)

    for _e in net.edges:
        for i in range(net.get_num_nodes):
            for _arc in reduced_directed.edges:
                _edge_of_arch = _arc
                j = _arc[0]
                k = _arc[1]
                if _edge_of_arch not in list(net.edges):
                    _edge_of_arch = (_arc[1], _arc[0])
                    assert _edge_of_arch in net.edges

                m.add_constraint((pi_edges_dict[_e][_edge_of_arch] + pe_edges_dict[_e][(i, j)] - pe_edges_dict[_e][(i, k)]) >= 0)

    logger.info("LP: Solving")
    m.solve()
    if logger.level == logging.DEBUG:
        m.print_solution()

    per_edge_flow_fraction = dict()
    for _edge in net.edges:
        edge_per_demands = np.zeros((net.get_num_nodes, net.get_num_nodes))
        _arch = _edge
        _reversed_arch = (_edge[1], _edge[0])
        for src, dst in net.get_all_pairs():
            assert src != dst
            edge_per_demands[src, dst] += f_arch_dict[_arch][(src, dst)].solution_value + f_arch_dict[_reversed_arch][
                (src, dst)].solution_value

        per_edge_flow_fraction[_edge] = edge_per_demands
    return r.solution_value, per_edge_flow_fraction


net = NetworkClass(topology_zoo_loader("http://www.topology-zoo.org/files/Ibm.gml", default_capacity=45))
oblivious_routing(net)
