"""
Created on 10 Mar 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""
from consts import EdgeConsts
import numpy as np
import networkx as nx
from collections import defaultdict
from utils import logger


class ECMPNetwork:

    def __init__(self, graph, min_weight=1e-12, max_weight=50):
        logger.info("Creating ECMP network")
        self._graph = graph.copy()
        self._graph_original = self._graph.copy()

        self._min_weight = min_weight  # min_weight
        self._max_weight = max_weight

        self._all_edges = self._graph.edges()

        self._capacities = None  # this is a vector of capacities
        self._c_nodes = None  # this is a vector of outgoing capacities for each node
        self._all_pairs = None
        self._num_nodes = len(self._graph.nodes())
        self._num_edges = len(self._graph.edges())
        self._actual_weight = np.zeros(self._graph.number_of_edges())
        self._set_adjacency()  # mark all adjacent nodes

    def _set_adjacency(self):
        logger.debug("Set adjacent node indicators")
        self._adj = np.zeros((self._num_nodes, self._num_nodes), dtype=np.float32)
        for i in self._graph.nodes:
            for j in self._graph.nodes:
                if i == j:
                    continue
                if j in self._graph[i]:
                    self._adj[i, j] = 1.0

    @property
    def get_adjacency(self):
        return self._adj

    def get_capacities(self):
        if self._capacities is None:  # for happens only once
            logger.debug("Set per edge capacity")
            self._capacities = dict()
            for i in self._graph.nodes:
                for j in self._graph.nodes:
                    if i == j:
                        continue
                    if self.get_adjacency[i, j]:  # means edge is exists
                        self._capacities[(i, j)] = self._graph[i][j][EdgeConsts.CAPACITY_STR]
        return self._capacities

    def get_node_capacities(self):
        if self._c_nodes is None:  # for happens only once
            logger.debug("Set per node capacity")
            self._c_nodes = defaultdict(int)
            for i in self._graph.nodes:
                for j in self._graph.nodes:
                    if i == j:
                        continue
                    if self.get_adjacency[i, j]:
                        self._c_nodes[i] += self._graph[i][j][EdgeConsts.CAPACITY_STR]
        return self._c_nodes

    @property
    def get_num_nodes(self):
        return self._num_nodes

    @property
    def get_num_edges(self):
        return self._num_edges

    @property
    def get_graph(self):
        return self._graph

    def get_all_pairs(self):
        """ return all nodes pairs"""
        if self._all_pairs is None:
            logger.debug("build all node pair list")
            self._all_pairs = []
            for src_ind in self._graph.nodes:
                for dst_ind in self._graph.nodes:
                    if dst_ind == src_ind:
                        continue
                    self._all_pairs.append((src_ind, dst_ind))
        return self._all_pairs

    def __getitem__(self, item):
        return self._graph[item]

    def all_simple_paths(self, source, target):
        return nx.all_simple_paths(self.get_graph, source=source, target=target)

    def get_edge_key(self, edge, key):
        return self.get_graph.edges[edge][key]

# def get_base_graph():
#     # init a triangle if we don't get a network graph
#     g = nx.Graph()
#     g.add_nodes_from([0, 1, 2])
#     g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
#                       (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
#                       (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 15})])
#
#     return g
#
#
# ecmpNetwork = ECMPNetwork(get_base_graph())
# adj = ecmpNetwork.get_adjacency
# pairs = ecmpNetwork.get_all_pairs()
# capcities = ecmpNetwork.get_capacities()
# node_capcities = ecmpNetwork.get_node_capacities()
# pass
