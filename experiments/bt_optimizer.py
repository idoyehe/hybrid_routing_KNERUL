from common.network_class import NetworkClass, nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from common.logger import *
from common.RL_Env.optimizer_abstract import *
from math import fsum
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_baseline_solver


class BTOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, testing=False):
        super(BTOptimizer, self).__init__(net, testing)
        self.static_splitting_ratios = np.zeros(shape=(net.get_num_nodes, net.get_num_edges), dtype=np.float)
        for u in net.nodes():
            fraction = 1 / len(net.out_edges_by_node(u))
            for _, v in net.out_edges_by_node(u):
                edge_id = net.get_edge2id(u, v)
                self.static_splitting_ratios[u, edge_id] = fraction

    def step(self, actions, traffic_matrix, optimal_value):
        _, splitting_ratios_per_src_dst_edge, _, _ = multiple_matrices_mcf_LP_baseline_solver(self._network, [(1.0, traffic_matrix)])
        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
            self._calculating_traffic_distribution(splitting_ratios_per_src_dst_edge, traffic_matrix)

        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    def _calculating_traffic_distribution(self, splitting_ratios, tm):
        net_direct = self._network


if __name__ == "__main__":
    from common.topologies import topology_zoo_loader

    net = NetworkClass(topology_zoo_loader(url="C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\common\\graphs_gmls\\T-lex.txt"))

    opt = BTOptimizer(net)
    pass
