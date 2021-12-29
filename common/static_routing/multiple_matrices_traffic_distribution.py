from common.utils import extract_flows
from common.network_class import NetworkClass
import numpy as np
import numpy.linalg as npl
import warnings


def multiple_matrices_traffic_distribution(net_direct: NetworkClass, traffic_matrices_list, src_dst_splitting_ratios):
    expected_objective, r_per_mtrx = _aux_multiple_tms_mcf_algebraic_solver(net_direct, traffic_matrices_list, src_dst_splitting_ratios)
    return expected_objective, r_per_mtrx


def _aux_multiple_tms_mcf_algebraic_solver(net_direct, traffic_matrices_list, src_dst_splitting_ratios):
    """Preparation"""
    tms_list_length = len(traffic_matrices_list)
    demands_ratios = np.zeros(shape=(tms_list_length, net_direct.get_num_nodes, net_direct.get_num_nodes))
    bt_per_mtrx = np.zeros(shape=(tms_list_length))
    aggregate_tm = np.sum(traffic_matrices_list, axis=0)
    active_flows = extract_flows(aggregate_tm)

    # extracting demands ratios per single matrix
    for m_idx, tm in enumerate(traffic_matrices_list):
        for src, dst in active_flows:
            assert aggregate_tm[src, dst] > 0
            demands_ratios[m_idx, src, dst] = tm[src, dst] / aggregate_tm[src, dst]
            assert 0 <= demands_ratios[m_idx, src, dst] <= 1

    flows_src2dest_per_node = dict()
    for src, dst in active_flows:
        psi = src_dst_splitting_ratios[(src, dst)]
        demand = np.zeros(shape=(net_direct.get_num_nodes))
        demand[src] = aggregate_tm[src, dst]
        assert all(psi[dst][:] == 0)
        assert psi.shape == (net_direct.get_num_nodes, net_direct.get_num_nodes)
        try:
            flows_src2dest_per_node[(src, dst)] = demand @ npl.inv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)
        except npl.LinAlgError as e:
            warnings.warn(str(e))
            flows_src2dest_per_node[(src, dst)] = demand @ npl.pinv(np.identity(net_direct.get_num_nodes, dtype=np.float64) - psi)
    for tm_idx in range(tms_list_length):
        tm_total_load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)
        for u, v in net_direct.edges:
            edge_idx = net_direct.get_edge2id(u, v)
            tm_total_load_per_link[edge_idx] = sum(
                (flows_src2dest_per_node[(src, dst)] * demands_ratios[tm_idx, src, dst])[u] * src_dst_splitting_ratios[src, dst][u, v] for src, dst in
                active_flows)

        tm_congestion = tm_total_load_per_link / net_direct.get_edges_capacities()
        bt_per_mtrx[tm_idx] = max(tm_congestion)

    expected_congestion = np.mean(bt_per_mtrx)

    return expected_congestion, bt_per_mtrx
