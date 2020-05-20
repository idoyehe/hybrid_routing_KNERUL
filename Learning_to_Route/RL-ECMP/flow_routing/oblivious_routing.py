from Learning_to_Route.data_generation.tm_generation import one_sample_tm_base
from Learning_to_Route.common.consts import Consts
from flow_routing.find_optimal_load_balancing import *
from network_class import NetworkClass, EdgeConsts
from topologies import topologies, topology_zoo_loader
from docplex.mp.model import Model


def _set_polyhedron_H1_constratins(net: NetworkClass, lp_edge):
    m = Model(name='Lp for oblivious routing with edge')
    traffic_demands = [
        [m.continuous_var(name="D_{}_{}".format(i, j), lb=0) if i != j else 0 for j in range(net.get_num_nodes)]
        for i in range(net.get_num_nodes)]

    edges_vars_dict = dict()
    reduced_directed, edge_map_dict = net.reducing_undirected2directed()

    out_edge_dict = defaultdict(list)
    in_edge_dict = defaultdict(list)

    for u, v, capacity in reduced_directed.edges.data(EdgeConsts.CAPACITY_STR):
        _edge = (u, v)
        out_edge_dict[u].append(_edge)
        in_edge_dict[v].append(_edge)

        edges_vars_dict[_edge] = []
        _all_edge_vars = []
        for i in range(net.get_num_nodes):
            from_i2j_demends = []
            for j in range(net.get_num_nodes):
                if i == j:
                    from_i2j_demends.append(0)
                    continue
                g_var = m.continuous_var(name="{}_g_{}_{}".format(str(_edge), i, j), lb=0)
                from_i2j_demends.append(g_var)
                _all_edge_vars.append(g_var)
            edges_vars_dict[_edge].append(from_i2j_demends)
        if capacity != float("inf"):
            m.add_constraint(m.sum(_all_edge_vars) <= capacity)
        if (u, v) == lp_edge:
            m.maximize(m.sum(_all_edge_vars) / capacity)

    for i in net.get_graph.nodes:  # iterate over original node only
        for j in net.get_graph.nodes:
            if i == j:
                continue
            out_g_i_j = [edges_vars_dict[out_edge][i][j] for out_edge in out_edge_dict[i]]
            in_g_i_j = [edges_vars_dict[in_edge][i][j] for in_edge in in_edge_dict[i]]
            m.add_constraint(m.sum(out_g_i_j) - m.sum(in_g_i_j) == traffic_demands[i][j])

            in_g_i_j = [edges_vars_dict[in_edge][j][i] for in_edge in in_edge_dict[i]]
            out_g_i_j = [edges_vars_dict[out_edge][j][i ] for out_edge in out_edge_dict[i]]
            m.add_constraint(m.sum(in_g_i_j) - m.sum(out_g_i_j) == traffic_demands[j][i])

    for k in reduced_directed.nodes:  # iterate over all nodes
        for i in net.get_graph.nodes:
            if i == k:
                continue
            for j in net.get_graph.nodes:
                if i == j or j == k:
                    continue
                out_g_i_j = [edges_vars_dict[out_edge][i][j] for out_edge in out_edge_dict[k]]
                in_g_i_j = [edges_vars_dict[in_edge][i][j] for in_edge in in_edge_dict[k]]
                m.add_constraint(m.sum(out_g_i_j) - m.sum(in_g_i_j) == 0)

    m.solve()
    m.print_solution()
    for i in net.get_graph.nodes:  # iterate over original node only
        for j in net.get_graph.nodes:
            if i == j:
                t_val = 0
            else:
                t_val = traffic_demands[i][j].solution_value
            print("D_{}_{}={}".format(i, j, t_val))

    return traffic_demands


ecmpNetwork = NetworkClass(topologies["TRIANGLE"])
# ecmpNetwork = ECMPNetwork(topology_zoo_loader("http://www.topology-zoo.org/files/Ibm.gml", default_capacity=45))
lp_edge = ecmpNetwork.reducing_undirected2directed()[1][(0, 1)]
_set_polyhedron_H1_constratins(ecmpNetwork, lp_edge)
