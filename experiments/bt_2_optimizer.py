from common.network_class import NetworkClass, nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from common.logger import *
from common.RL_Env.optimizer_abstract import *
from math import fsum
from sys import argv
from argparse import ArgumentParser
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file, extract_flows
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_baseline_solver
from common.RL_Env.rl_env_consts import HistoryConsts


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


class BT2Optimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, actions, testing=False):
        super(BT2Optimizer, self).__init__(net, testing)
        self._actions = actions

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        weights_splitting_ratios = self._calculating_splitting_ratios(weights_vector)
        _, optimal_splitting_ratios_per_src_dst_edge, _, _ = multiple_matrices_mcf_LP_baseline_solver(self._network, [(1.0, traffic_matrix)])

        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
            self.__calculating_traffic_distribution(weights_splitting_ratios, optimal_splitting_ratios_per_src_dst_edge, traffic_matrix)

        return total_congestion, max_congestion, total_load_per_link, most_congested_link

    def _calculating_splitting_ratios(self, weights_vector):
        net_direct = self._network
        splitting_ratios = np.zeros((net_direct.get_num_nodes, net_direct.get_num_edges), dtype=np.float64)
        one_hop_cost = (weights_vector * self._outgoing_edges) @ np.transpose(self._ingoing_edges)

        reduced_directed_graph = nx.DiGraph()
        for edge_index, cost in enumerate(weights_vector):
            u, v = net_direct.get_id2edge(edge_index)
            reduced_directed_graph.add_edge(u, v, cost=cost)

        for node_dst in net_direct.nodes:
            cost_adj = nx.shortest_path_length(G=reduced_directed_graph, target=node_dst, weight='cost')
            cost_adj = [cost_adj[i] for i in range(self._num_nodes)]
            edge_cost = self.__get_edge_cost(cost_adj, one_hop_cost)
            q_val = self._soft_min(edge_cost)
            splitting_ratios[node_dst] = q_val

        return splitting_ratios

    def _soft_min(self, weights_vector, alpha=HistoryConsts.SOFTMIN_ALPHA):
        """
        :param weights_vector: vector of weights
        :param alpha: for exponent expression
        :return: sum over deges
        """

        exp_val = np.exp(alpha * weights_vector)
        exp_val[weights_vector == 0] = 0
        exp_val[np.logical_and(weights_vector != 0, exp_val == 0)] = HistoryConsts.EPSILON

        exp_val = np.transpose(exp_val) / np.sum(exp_val, axis=1)
        exp_val = np.sum(np.transpose(exp_val), axis=0)
        net_direct = self._network
        for u in net_direct.nodes:
            error_bound(1.0, sum(exp_val[net_direct.get_edge2id(u, v)] for _, v in net_direct.out_edges_by_node(u)))
        return exp_val

    def __calculating_traffic_distribution(self, weights_splitting_ratios, optimal_splitting_ratios_per_src_dst_edge, tm):
        net_direct = self._network
        total_load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)
        for src, dst in extract_flows(tm):
            demand = tm[src, dst]
            total_load_per_link += self._simulate_demand(weights_splitting_ratios, optimal_splitting_ratios_per_src_dst_edge, src, dst, demand)

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)

        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    def _simulate_demand(self, weights_splitting_ratios, optimal_splitting_ratios_per_src_dst_edge, src, dst, demand):
        net_direct = self._network
        gb_env = gb.Env(empty=True)
        gb_env.setParam(GRB.Param.OutputFlag, 0)
        gb_env.setParam(GRB.Param.NumericFocus, 2)
        gb_env.setParam(GRB.Param.FeasibilityTol, 1e-6)
        gb_env.start()

        lp_problem = gb.Model(name="LP problem for flow, given network, source, destination and splitting_ratios",
                              env=gb_env)
        flows_vars_per_edge = lp_problem.addVars(net_direct.edges, name="f", lb=0.0, vtype=GRB.CONTINUOUS)

        for u in net_direct.nodes:
            if u == dst:
                lp_problem.addConstrs((flows_vars_per_edge[arch] == 0 for arch in net_direct.out_edges_by_node(dst)),
                                      name="dst_{}_out_links".format(dst))
                _collected_flow_in_t_destined_t = sum(flows_vars_per_edge[arch] for arch in net_direct.in_edges_by_node(dst))
                lp_problem.addConstr(_collected_flow_in_t_destined_t == demand, name="dst_{}_in_links".format(dst))
                continue

                # all incoming with originated from s to t
            _collected_flow_in_u_destined_t = sum(flows_vars_per_edge[arch] for arch in net_direct.in_edges_by_node(u))
            if u == src:
                _collected_flow_in_u_destined_t += demand

            _outgoing_flow_from_u_destined_t = sum(flows_vars_per_edge[arch] for arch in net_direct.out_edges_by_node(u))
            lp_problem.addConstr(_collected_flow_in_u_destined_t == _outgoing_flow_from_u_destined_t, name="flow_{}_to_{}".format(u, dst))

            for out_arch in net_direct.out_edges_by_node(u):
                edge_index = net_direct.get_edge2id(*out_arch)

                if self._actions[u] == 0:
                    lp_problem.addConstr(
                        flows_vars_per_edge[out_arch] == _collected_flow_in_u_destined_t * weights_splitting_ratios[dst, edge_index],
                        name="dst_{}_sr_({},{})".format(dst, *out_arch))

                else:
                    assert self._actions[u] == 1
                    spr = optimal_splitting_ratios_per_src_dst_edge[(src, dst) + out_arch]
                    if spr is None:
                        spr = 0
                    lp_problem.addConstr(
                        flows_vars_per_edge[out_arch] == _collected_flow_in_u_destined_t * spr, name="dst_{}_sr_({},{})".format(dst, *out_arch))

        lp_problem.update()
        try:
            logger.debug("LP Submit to Solve {}".format(lp_problem.ModelName))
            lp_problem.optimize()
            assert lp_problem.Status == GRB.OPTIMAL
        except AssertionError as e:
            logger_level = logger.level
            logger.setLevel(logging.DEBUG)
            if lp_problem.Status == GRB.UNBOUNDED:
                logger.debug('The model cannot be solved because it is unbounded')
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

            if lp_problem.Status != GRB.INF_OR_UNBD and lp_problem.Status != GRB.INFEASIBLE:
                logger.debug('Optimization was stopped with status {}'.format(lp_problem.Status))
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

            orignumvars = lp_problem.NumVars
            lp_problem.feasRelaxS(0, False, False, True)
            lp_problem.optimize()

            if lp_problem.Status != GRB.OPTIMAL:
                raise Exception("****Optimize failed****\nStatus is NOT optimal but {}".format(lp_problem.Status))

            assert logger.level == logging.DEBUG
            logger.debug('\nSlack values:')
            slacks = lp_problem.getVars()[orignumvars:]
            for sv in slacks:
                if sv.X > 1e-6:
                    logger.debug('{} = {}'.format(sv.VarName, sv.X))

            logger.setLevel(logger_level)


        except gb.GurobiError as e:
            raise Exception("****Optimize failed****\nException is:\n{}".format(e))

        if logger.level == logging.DEBUG:
            lp_problem.printStats()
            lp_problem.printQuality()

        flows_vars_per_edge = dict(flows_vars_per_edge)
        for key in flows_vars_per_edge.keys():
            flows_vars_per_edge[key] = flows_vars_per_edge[key].x

        lp_problem.close()
        gb_env.close()

        # self.__validate_flow(net_direct, tm, flows_vars_per_per_dest_per_edge, splitting_ratios)

        total_load_per_link = np.zeros((net_direct.get_num_edges), dtype=np.float64)

        for u, v in net_direct.edges:
            edge_index = net_direct.get_edge2id(u, v)
            total_load_per_link[edge_index] = flows_vars_per_edge[(u, v)]

        return total_load_per_link

    def __get_edge_cost(self, cost_adj, each_edge_weight):
        cost_to_dst1 = cost_adj * self._graph_adjacency_matrix + each_edge_weight
        cost_to_dst2 = np.reshape(cost_to_dst1, [-1])
        cost_to_dst3 = cost_to_dst2[cost_to_dst2 != 0]
        return cost_to_dst3 * self._outgoing_edges


if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))
    actions = np.zeros(shape=(net.get_num_nodes), dtype=np.int)
    actions[3] = actions[11] = 1
    opt = BT2Optimizer(net, actions)
    opt.step(np.ones(shape=(net.get_num_edges)), loaded_dict["tms"][0][0], loaded_dict["tms"][0][1])
