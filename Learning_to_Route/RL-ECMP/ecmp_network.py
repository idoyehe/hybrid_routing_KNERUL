"""
Created on 10 Mar 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""
from consts import EdgeConsts
import numpy as np


class ECMPNetwork:

    def __init__(self, graph, weight_func=None, max_weight=50, epsilon=1e-12):
        self._graph = graph.copy()
        self._graph_original = self._graph.copy()

        self._weight_func = weight_func

        self._min_weight = epsilon  # min_weight
        self._max_weight = max_weight
        self._epsilon = epsilon

        self._flows = []

        self._all_edges = self._graph.edges()
        self._all_ids = {e: eid for eid, e in enumerate(self._all_edges)}

        self._c = None  # this is a vector of capacities
        self._c_nodes = None  # this is a vector of outgoing capacities for each node
        self._num_nodes = len(self._graph.nodes())
        self._num_edges = len(self._graph.edges())
        self._actual_weight = np.zeros(self._graph.number_of_edges())
        self._set_adjacency()  # mark all adjacent nodes

    def _set_adjacency(self):
        self._adj = np.zeros((self._num_nodes, self._num_nodes), dtype=np.float32)
        for i in range(self._num_nodes):
            for j in range(self._num_nodes):
                if i == j:
                    continue
                if j in self._graph[i]:
                    self._adj[i, j] = 1.0

    @property
    def get_adjacency(self):
        return self._adj

    def get_capacities(self):
        if self._c is None:  # for happens only once
            eid = 0
            c = np.zeros(2 * self._num_edges, dtype=np.float32)
            for i in range(self._num_nodes):
                for j in range(self._num_nodes):
                    if i == j:
                        continue
                    if self.get_adjacency[i, j]:  # means edge is exists
                        c[eid] = self._graph[i][j][EdgeConsts.CAPACITY_STR]
                        eid += 1
            self._c = c
        return self._c


    def get_node_capacities(self):
        if self._c_nodes is None:  # for happens only once
            c = np.zeros((self._num_nodes,), dtype=np.float32)
            for i in range(self._num_nodes):
                for j in range(self._num_nodes):
                    if i == j:
                        continue
                    if self.get_adjacency[i, j]:
                        c[i] += self._graph[i][j][EdgeConsts.CAPACITY_STR]
            self._c_nodes = c

        return self._c_nodes


    @property
    def get_all_edges_id(self):
        return self._all_ids


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
        nodes = list(self._graph.nodes())
        pairs = []
        for src_ind in range(len(nodes)):
            src = nodes[src_ind]
            for dst_ind in range(len(nodes)):
                if dst_ind == src_ind:
                    continue
                dst = nodes[dst_ind]
                pairs.append((src, dst))
        return pairs

    def __getitem__(self, item):
        return self._graph[item]
