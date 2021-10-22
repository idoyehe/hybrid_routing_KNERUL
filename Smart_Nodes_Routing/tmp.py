import torch
import torch.optim as optim
import torch.nn as nn
from Link_State_Routing_PEFT.RL.PEFT_optimizer import PEFTOptimizer
from Learning_to_Route.rl_softmin_history.soft_min_optimizer import SoftMinOptimizer
import json
from common.topologies import topology_zoo_loader
from common.network_class import NetworkClass
from common.utils import load_dump_file
import numpy as np
import matplotlib.pyplot as plt
from Link_State_Routing_PEFT.gradiant_decent.original_PEFT import PEFT_main_loop
from functools import partial
from multiprocessing import Pool
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def clac_grad_of_i(i, traffic_distribution, traffic_matrix_list, epsilon, w_u_v):
    w_right = np.copy(w_u_v)
    w_right[i] += epsilon

    w_left = np.copy(w_u_v)
    w_left[i] -= epsilon

    dst_spr_right = traffic_distribution.calculating_destination_based_spr(w_right)
    dst_spr_left = traffic_distribution.calculating_destination_based_spr(w_left)

    right = 0
    left = 0
    for tm in traffic_matrix_list:
        right += traffic_distribution._calculating_traffic_distribution(dst_spr_right, tm)[0]
        left += traffic_distribution._calculating_traffic_distribution(dst_spr_left, tm)[0]

    prob = 1 / len(traffic_matrix_list)

    return ((right - left) * prob) / (2 * epsilon)


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, net_direct, w_u_v, traffic_matrix_list, epsilon):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.tensor(w_u_v, device=device, requires_grad=True)
        # make weights torch parameters
        self.weights = nn.Parameter(weights)
        self.traffic_matrix_list = traffic_matrix_list
        self.net_direct = net_direct
        self.traffic_distribution = SoftMinOptimizer(self.net_direct,-1)
        self.prob = 1 / len(self.traffic_matrix_list)
        self.epsilon = epsilon

        dst_splitting_ratios = self.traffic_distribution.calculating_destination_based_spr(w_u_v)
        expected_congestion = np.mean(
            [self.traffic_distribution._calculating_traffic_distribution(dst_splitting_ratios, tm)[0] for tm in traffic_matrix_list])
        print("Expected Congestion: {}".format(expected_congestion))

    def forward(self):
        dsr_spr = self.traffic_distribution.calculating_destination_based_spr(self.weights.detach().numpy())
        expected_congestion = 0
        for tm in self.traffic_matrix_list:
            expected_congestion += self.traffic_distribution._calculating_traffic_distribution(dsr_spr, tm)[0]
        expected_congestion *= self.prob
        return expected_congestion

    def backward(self):

        self.weights.grad = torch.zeros_like(self.weights, device=device, requires_grad=False)

        random.shuffle(self.traffic_matrix_list)
        current_traffic_matrix_list = self.traffic_matrix_list[0:8]

        clac_grad_of_i_wrapper = partial(clac_grad_of_i, traffic_distribution=self.traffic_distribution, traffic_matrix_list=current_traffic_matrix_list,
                                         epsilon=self.epsilon, w_u_v=self.weights.detach().numpy())
        processes = 4
        start_idx = 0
        evaluations = list()
        while start_idx < self.net_direct.get_num_edges:
            end_idx = min(start_idx + processes, self.net_direct.get_num_edges)
            pool = Pool(processes=end_idx - start_idx)
            evaluations += pool.map(func=clac_grad_of_i_wrapper, iterable=list(range(start_idx, end_idx)))
            pool.terminate()
            start_idx += processes

        for i, grad_val in enumerate(evaluations):
            self.weights.grad[i] = grad_val


def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        print("Iteration: {}".format(i + 1))
        model.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(m.weights.detach().numpy())
        loss = m.forward()
        losses.append(loss)
        print("current loss: {}".format(loss))
    return losses


if __name__ == "__main__":
    train_file = "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\common\\TMs_DB\\ScaleFree30Nodes\\ScaleFree30Nodes_tms_30X30_length_32_gravity_sparsity_0.3"
    loaded_dict = load_dump_file(train_file)
    traffic_matrix_list = [t[0] for t in loaded_dict['tms']]
    net = NetworkClass(topology_zoo_loader(loaded_dict["url"]))
    traffic_matrix = sum(traffic_matrix_list)
    necessary_capacity = loaded_dict["necessary_capacity_per_tm"][-1]
    # w_u_v, PEFT_congestion = PEFT_main_loop(net, traffic_matrix, necessary_capacity, 1)
    w_u_v = [3.47412, 4.11262, 2.24398, 2.46311, 2.16503, 3.05864, 2.95789, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 2.28427, 3.35125,
             3.26659, 3.59131, 2.19259, 1.99755, 10.00000, 2.44873, 10.00000, 10.00000, 4.05422, 3.17143, 3.28335, 2.95099, 10.00000, 10.00000,
             2.07912, 2.36841, 10.00000, 10.00000, 3.55714, 3.25561, 2.41069, 10.00000, 2.65612, 2.39013, 2.74614, 2.57212, 10.00000, 10.00000,
             10.00000, 2.32439, 2.33028, 10.00000, 2.52557, 2.14435, 2.64191, 2.10648, 2.05611, 2.40190, 1.14525, 3.09287, 2.82856, 2.75649, 1.88927,
             10.00000, 10.00000, 10.00000, 2.07669, 2.58272, 1.76219, 2.44488, 2.36743, 1.77035, 1.71749, 10.00000, 3.02397, 1.81141, 1.45571,
             10.00000, 1.15702, 1.44394, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 2.28092, 1.72084, 10.00000,
             10.00000]

    m = Model(net, w_u_v, traffic_matrix_list, 0.0001)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)

    losses = training_loop(m, opt)
    plt.figure(figsize=(14, 7))
    plt.plot(losses)
    # print(m.weights)
