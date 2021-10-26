from Learning_to_Route.rl_softmin_history.soft_min_optimizer import SoftMinOptimizer
from common.RL_Envs.optimizer_abstract import *
from common.utils import extract_flows
from common.consts import EdgeConsts
import numpy.linalg as npl


class SmartNodesOptimizer(SoftMinOptimizer):
    def __init__(self, net: NetworkClass, testing=False):
        super(SmartNodesOptimizer, self).__init__(net, -1, testing)

    def step(self, weights_vector, traffic_matrix, optimal_value):
        """
        :param weights_vector: the weights vector per edge from agent
        :param traffic_matrix: the traffic matrix to examine
        :return: cost and congestion
        """
        max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link = self._get_cost_given_weights(weights_vector,
                                                                                                                                       traffic_matrix,
                                                                                                                                       optimal_value)

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link

    def _get_cost_given_weights(self, links_weights, traffic_matrix, optimal_value):
        net_direct = self._network
        dst_splitting_ratios = self.calculating_destination_based_spr(links_weights)

        if len(net_direct.get_smart_nodes) > 0:
            src_dst_splitting_ratios = self.calculating_src_dst_spr(dst_splitting_ratios)
            max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, \
            load_per_link = self._calculating_traffic_distribution(src_dst_splitting_ratios, traffic_matrix)
        else:
            assert len(net_direct.get_smart_nodes) == 0
            max_congestion, \
            most_congested_link, \
            flows_to_dest_per_node, \
            congestion_per_link, \
            load_per_link = super(SmartNodesOptimizer, self)._calculating_traffic_distribution(dst_splitting_ratios, traffic_matrix)

        if self._testing:
            logger.info("RL most congested link: {}".format(most_congested_link))
            logger.info("RL MLU: {}".format(max_congestion))
            logger.info("RL MLU Vs. Optimal: {}".format(max_congestion / optimal_value))

        return max_congestion, most_congested_link, flows_to_dest_per_node, congestion_per_link, load_per_link

    def _calculating_traffic_distribution(self, src_dst_splitting_ratios, traffic_matrix):
        return super(SmartNodesOptimizer, self)._calculating_src_dst_traffic_distribution(src_dst_splitting_ratios, traffic_matrix)
