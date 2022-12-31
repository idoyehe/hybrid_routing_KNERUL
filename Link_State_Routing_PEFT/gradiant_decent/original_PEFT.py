import torch
import torch.nn as nn
from argparse import ArgumentParser
from common.static_routing.optimal_load_balancing import optimal_load_balancing_LP_solver
from common.network_class import NetworkClass
from common.topologies import topology_zoo_loader
from common.logger import logger
from common.consts import DumpsConsts
from common.utils import load_dump_file
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
from Learning_to_Route.rl_softmin_history.soft_min_optimizer import SoftMinOptimizer
from sys import argv
import numpy as np
from tabulate import tabulate


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for dump file")
    parser.add_argument("-p", "--dumped_path", type=str, help="The path for the dumped file")
    options = parser.parse_args(args)
    return options


class PEFT_Model(nn.Module):
    """Custom Pytorch model for PEFT gradient optimization.
    """

    def __init__(self, network: NetworkClass, traffic_matrix, factor=10):
        super().__init__()
        self._network: NetworkClass = network
        # initialize weights with random numbers
        weights = self.__initialize_all_weights(factor)
        # make weights torch parameters
        self._weights = nn.Parameter(weights)
        self._traffic_matrix = traffic_matrix
        self._softMin = SoftMinOptimizer(self._network, -1)
        self._peft_traffic = PEFTOptimizer(self._network)
        self._peft_current_flows_values = None
        self._set_necessary_capacity()
        self._lr = 1 / np.max(self._necessary_capacity)
        self.forward()

    def _set_necessary_capacity(self):
        self._optimal_value, self._necessary_capacity = optimal_load_balancing_LP_solver(self._network,
                                                                                         self._traffic_matrix)

    def __initialize_all_weights(self, factor):
        return torch.ones(size=(self._network.get_num_edges,), dtype=torch.float64, device="cpu", requires_grad=True) * factor

    def forward(self):
        PEFT_congestion, _, _, _, self._peft_current_flows_values = self._peft_traffic.step(
            self._weights.detach().numpy(), self._traffic_matrix,
            self._optimal_value)
        softMin_congestion = \
            self._softMin.step(self._weights.detach().numpy(), self._traffic_matrix, self._optimal_value)[0]
        logger.info('PEFT Congestion Vs. Optimal= {}'.format(PEFT_congestion / self._optimal_value))
        logger.info('SoftMin Congestion Vs. Optimal= {}'.format(softMin_congestion / self._optimal_value))

    def backward(self):
        self._weights.grad = torch.zeros_like(self._weights, device="cpu", requires_grad=False)
        for u, v in self._network.edges:
            edge_idx = self._network.get_edge2id(u, v)
            self._weights.grad[edge_idx] = self._necessary_capacity[u, v] - self._peft_current_flows_values[edge_idx]

    def loss(self):
        net = self._network
        delta = sum(
            np.abs(self._peft_current_flows_values[net.get_edge2id(u, v)] - self._necessary_capacity[u, v]) for u, v in
            net.edges)
        logger.info('loss= {}'.format(delta))
        return delta

    @property
    def get_learning_rate(self):
        return self._lr

    @property
    def get_weights(self):
        return self._weights.detach().numpy()

    def step(self):
        new_weights = torch.zeros_like(self._weights)
        for u, v in self._network.edges:
            edge_idx = self._network.get_edge2id(u, v)
            new_weights[edge_idx] = max(0,self._weights[edge_idx] - self._lr * self._weights.grad[edge_idx])

        self._weights = nn.Parameter(new_weights)


class PEFT_Model_With_Init(PEFT_Model):
    def __init__(self, network: NetworkClass, traffic_matrix, necessary_capacity, opt_value, factor=10):
        self._optimal_value = opt_value
        self._necessary_capacity = necessary_capacity
        super(PEFT_Model_With_Init, self).__init__(network, traffic_matrix, factor)

    def _set_necessary_capacity(self):
        pass


def PEFT_training_loop(net, traffic_matrix, stop_threshold=0.5, factor=10):
    peft_model = PEFT_Model(net, traffic_matrix, factor)
    while peft_model.loss() > stop_threshold:
        peft_model.backward()
        peft_model.step()
        peft_model.zero_grad()
        peft_model.forward()

    return peft_model.get_weights


def PEFT_training_loop_with_init(net, traffic_matrix, necessary_capacity, opt_value=1.0, stop_threshold=0.5, factor=10):
    peft_model = PEFT_Model_With_Init(net, traffic_matrix, necessary_capacity, opt_value, factor)
    # opt = torch.optim.SGD(peft_model.parameters(), lr=peft_model.get_learning_rate, momentum=0.9)
    while peft_model.loss() > stop_threshold:
        peft_model.backward()
        peft_model.step()
        peft_model.zero_grad()
        peft_model.forward()

    return peft_model.get_weights


if __name__ == "__main__":
    options = _getOptions()
    dumped_path = options.dumped_path
    loaded_dict = load_dump_file(dumped_path)
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))
    traffic = SoftMinOptimizer(net, -1)
    optimal_splitting_ratios = loaded_dict['optimal_splitting_ratios']

    N = len(np.array(list(zip(*loaded_dict["tms"]))[0]))
    traffic_matrix = sum(np.array(list(zip(*loaded_dict["tms"]))[0]) / N)
    total_load_per_link = traffic._calculating_src_dst_traffic_distribution(optimal_splitting_ratios, traffic_matrix)[4]

    necessary_capacity = np.zeros(shape=(net.get_num_nodes, net.get_num_nodes), dtype=np.float64)
    for u, v in net.edges:
        necessary_capacity[u, v] = total_load_per_link[net.get_edge2id(u, v)]

    weights = PEFT_training_loop_with_init(net, traffic_matrix, necessary_capacity, stop_threshold=1 / net.get_num_edges,factor=5)
    # print("Link Weights:\n{}".format(weights))
    loaded_dict[DumpsConsts.INITIAL_WEIGHTS] = weights

    import pickle

    dump_file = open(dumped_path, 'wb')
    pickle.dump(loaded_dict, dump_file)
    dump_file.close()
