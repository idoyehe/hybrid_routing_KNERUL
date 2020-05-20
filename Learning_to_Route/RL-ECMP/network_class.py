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
from logger import logger


class NetworkClass:

    def __init__(self, topo, min_weight=1e-12, max_weight=50):
        logger.info("Creating ECMP network")
        self._graph = topo.copy()
        self._is_directed = nx.is_directed(self.get_graph)

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
        self._g_directed = None
        self._reducing_map_dict = None

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

    @property
    def get_name(self):
        if "Name" in self.get_graph.graph:
            return self.get_graph.graph["Name"]
        return ""

    def get_edges_capacities(self):
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
        if self._is_directed:
            return self._num_edges
        return 2 * self._num_edges

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

    def all_simple_paths(self, source, target, cutoff=None):
        return nx.all_simple_paths(self.get_graph, source=source, target=target, cutoff=cutoff)

    def all_shortest_path(self, source, target, weight):
        """
        Parameters
        ----------
        G : NetworkX graph

        source : node
        Starting node for path.

        target : node
        Ending node for path.

        weight : None or string, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        """
        return nx.all_shortest_paths(self.get_graph, source=source, target=target, weight=weight)

    def get_edge_key(self, edge, key):
        return self.get_graph.edges[edge][key]

    def build_edges_map(self):
        """
        @return: num of edges, map for each node its ingoing and outgoing edges
        """
        graph_adjacency = self.get_adjacency
        num_edges = np.int32(np.sum(graph_adjacency))
        ingoing = np.zeros((graph_adjacency.shape[0], num_edges))
        outgoing = np.zeros((graph_adjacency.shape[0], num_edges))
        eid = 0
        e_map = {}
        for i in range(graph_adjacency.shape[0]):
            for j in range(graph_adjacency.shape[0]):
                if graph_adjacency[i, j] == 1:
                    outgoing[i, eid] = 1
                    ingoing[j, eid] = 1
                    e_map[(i, j)] = eid
                    eid += 1
        return num_edges, ingoing, outgoing, e_map

    def reducing_undirected2directed(self):
        if self._g_directed is None:
            self._g_directed = nx.DiGraph()
            self._g_directed.add_nodes_from(self._graph.nodes)
            current_node_index = self.get_num_nodes
            self._reducing_map_dict = dict()
            for (u, v, u_v_capacity) in self._graph.edges.data(EdgeConsts.CAPACITY_STR):
                x_index = current_node_index
                y_index = current_node_index + 1
                current_node_index += 2
                _virtual_edges_data = {EdgeConsts.WEIGHT_STR: 0, EdgeConsts.CAPACITY_STR: float("inf")}
                _reduced_edge_data = {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: u_v_capacity}

                self._g_directed.add_edge(u_of_edge=u, v_of_edge=x_index, **_virtual_edges_data)
                self._g_directed.add_edge(u_of_edge=v, v_of_edge=x_index, **_virtual_edges_data)
                self._g_directed.add_edge(u_of_edge=y_index, v_of_edge=u, **_virtual_edges_data)
                self._g_directed.add_edge(u_of_edge=y_index, v_of_edge=v, **_virtual_edges_data)

                self._g_directed.add_edge(u_of_edge=x_index, v_of_edge=y_index, **_reduced_edge_data)
                self._reducing_map_dict[(u, v)] = (x_index, y_index)
        return self._g_directed, self._reducing_map_dict

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
# ecmpNetwork.reducing_undirected2directed()
# adj = ecmpNetwork.get_adjacency
# pairs = ecmpNetwork.get_all_pairs()
# capcities = ecmpNetwork.get_capacities()
# node_capcities = ecmpNetwork.get_node_capacities()
# pass
