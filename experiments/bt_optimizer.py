from common.network_class import NetworkClass, nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from common.logger import *
from common.RL_Env.optimizer_abstract import *
from math import fsum
from Link_State_Routing_PEFT.MCF_problem.multiple_matrices_MCF import multiple_matrices_mcf_LP_baseline_solver
from sys import argv
from argparse import ArgumentParser
from common.topologies import topology_zoo_loader
from common.utils import load_dump_file, extract_flows


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


class BTOptimizer(Optimizer_Abstract):
    def __init__(self, net: NetworkClass, testing=False):
        super(BTOptimizer, self).__init__(net, testing)
        self.static_splitting_ratios = np.zeros(shape=(net.get_num_nodes, net.get_num_edges), dtype=np.float)
        for u in net.nodes():
            fraction = 1 / len(net.out_edges_by_node(u))
            for _, v in net.out_edges_by_node(u):
                edge_id = net.get_edge2id(u, v)
                self.static_splitting_ratios[u, edge_id] = fraction

        self._actions = None

    def step(self, actions, traffic_matrix, optimal_value):
        self._actions = actions
        _, splitting_ratios_per_src_dst_edge, _, _ = multiple_matrices_mcf_LP_baseline_solver(self._network, [(1.0, traffic_matrix)])

        max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link = \
            self._calculating_traffic_distribution(splitting_ratios_per_src_dst_edge, traffic_matrix)

        return total_congestion, max_congestion, total_load_per_link, most_congested_link

    def _calculating_traffic_distribution(self, splitting_ratios, tm):
        net_direct = self._network
        total_load_per_link = np.zeros(shape=(net_direct.get_num_edges), dtype=np.float64)
        for src, dst in extract_flows(tm):
            demand = tm[src, dst]
            total_load_per_link += self._simulate_demand(splitting_ratios, src, dst, demand)

        total_congestion_per_link = total_load_per_link / self._edges_capacities

        most_congested_link = np.argmax(total_congestion_per_link)
        max_congestion = total_congestion_per_link[most_congested_link]
        total_congestion = np.sum(total_congestion_per_link)

        return max_congestion, most_congested_link, total_congestion, total_congestion_per_link, total_load_per_link

    def _simulate_demand(self, splitting_ratios, src, dst, demand):
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
                        flows_vars_per_edge[out_arch] == _collected_flow_in_u_destined_t * self.static_splitting_ratios[u, edge_index],
                        name="dst_{}_sr_({},{})".format(dst, *out_arch))

                else:
                    assert self._actions[u] == 1
                    spr = splitting_ratios[(src, dst) + out_arch]
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












if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"], default_capacity=loaded_dict["capacity"]))

    opt = BTOptimizer(net)
    opt.step(np.ones(shape=(net.get_num_nodes)), loaded_dict["tms"][0][0], loaded_dict["tms"][0][1])