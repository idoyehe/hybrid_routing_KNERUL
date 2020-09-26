"""
Created on 10 Mar 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""
from common.consts import EdgeConsts
import numpy as np
import networkx as nx
from common.logger import logger
import matplotlib.pyplot as plt


class NetworkClass:

    def __init__(self, topo, min_weight=1e-12, max_weight=50):
        logger.info("Creating Network Class")

        self._graph = topo.copy()
        self._is_directed = nx.is_directed(self.get_graph)

        self._min_weight = min_weight  # min_weight
        self._max_weight = max_weight

        self._all_edges = self._graph.edges

        self._capacities = None  # this is a vector of capacities
        self._all_pairs = None
        self._num_nodes = len(self._graph.nodes)
        self._num_edges = len(self._all_edges)
        self._actual_weight = np.zeros(self._graph.number_of_edges())
        self._set_adjacency()  # mark all adjacent nodes
        self._g_directed_reduced = None
        self._reducing_map_dict = None
        self._id2edge_map = None
        self._edge2id_map = None

        if not self._is_directed:
            self._g_directed = NetworkClass(self.get_graph.to_directed())
        else:
            self._g_directed = self

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
    def get_g_directed(self):
        return self._g_directed

    @property
    def g_is_directed(self):
        return self._is_directed

    @property
    def get_adjacency(self):
        return self._adj

    @property
    def edges(self):
        return self.get_graph.edges

    @property
    def nodes(self):
        return self.get_graph.nodes

    @property
    def get_name(self):
        if "Name" in self.get_graph.graph:
            return self.get_graph.graph["Name"]
        return ""

    def get_edges_capacities(self):
        if self._capacities is None:  # for happens only once
            self.build_edges_map()
        return self._capacities

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
        assert len(self._all_pairs) == self.get_num_nodes * (self.get_num_nodes - 1)
        return self._all_pairs

    def __getitem__(self, item):
        return self._graph[item]

    def all_simple_paths(self, source, target, cutoff=None):
        return nx.all_simple_paths(self.get_graph, source=source, target=target, cutoff=cutoff)

    def all_shortest_path(self, source, target, weight=None, method='dijkstra'):
        """Compute all shortest paths in the graph.

        Parameters
        ----------
        self : NetworkX graph

        source : node
           Starting node for path.

        target : node
           Ending node for path.

        weight : None or string, optional (default = None)
           If None, every edge has weight/distance/cost 1.
           If a string, use this edge attribute as the edge weight.
           Any edge attribute not present defaults to 1.

        method : string, optional (default = 'dijkstra')
           The algorithm to use to compute the path lengths.
           Supported options: 'dijkstra', 'bellman-ford'.
           Other inputs produce a ValueError.
           If `weight` is None, unweighted graph methods are used, and this
           suggestion is ignored.

        Returns
        -------
        paths : generator of lists
            A generator of all paths between source and target.

        Raises
        ------
        ValueError
            If `method` is not among the supported options.

        NetworkXNoPath
            If `target` cannot be reached from `source`.
        """
        return nx.all_shortest_paths(self.get_graph, source=source, target=target, weight=weight, method=method)

    def get_edge_key(self, edge, key):
        return self.get_graph.edges[edge][key]

    def build_edges_map(self):
        """
        @return: num of edges, map for each node its ingoing and outgoing edges
        """

        graph_adjacency = self.get_adjacency
        num_edges = np.int32(np.sum(graph_adjacency))
        ingoing = np.zeros((len(graph_adjacency), num_edges))
        outgoing = np.zeros((len(graph_adjacency), num_edges))
        self._capacities = np.zeros(num_edges)
        self._id2edge_map = dict()
        self._edge2id_map = dict()
        eid = 0
        for i in range(len(graph_adjacency)):
            for j in range(len(graph_adjacency)):
                if graph_adjacency[i][j] == 1:
                    outgoing[i, eid] = 1
                    ingoing[j, eid] = 1
                    self._capacities[eid] = self.get_edge_key((i, j), EdgeConsts.CAPACITY_STR)
                    self._id2edge_map[eid] = (i, j)
                    self._edge2id_map[(i, j)] = eid
                    eid += 1
        return self._num_edges, ingoing, outgoing, self._capacities

    def get_id2edge(self):
        if self._id2edge_map is None:
            self.build_edges_map()
        return self._id2edge_map

    def get_edge2id(self):
        if self._edge2id_map is None:
            self.build_edges_map()
        return self._edge2id_map

    def reducing_undirected2directed(self):
        if self._g_directed_reduced is None:
            self._g_directed_reduced = nx.DiGraph()
            self._reducing_map_dict = dict()
            for (u, v, u_v_capacity) in self._graph.edges.data(EdgeConsts.CAPACITY_STR):
                x_node = (u, v, "in")
                y_node = (u, v, "out")
                _virtual_edges_data = {EdgeConsts.WEIGHT_STR: 0, EdgeConsts.CAPACITY_STR: u_v_capacity}
                _reduced_edge_data = {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: u_v_capacity}

                self._g_directed_reduced.add_edge(u_of_edge=u, v_of_edge=x_node, **_virtual_edges_data)
                self._g_directed_reduced.add_edge(u_of_edge=v, v_of_edge=x_node, **_virtual_edges_data)

                self._g_directed_reduced.add_edge(u_of_edge=y_node, v_of_edge=u, **_virtual_edges_data)
                self._g_directed_reduced.add_edge(u_of_edge=y_node, v_of_edge=v, **_virtual_edges_data)

                self._g_directed_reduced.add_edge(u_of_edge=x_node, v_of_edge=y_node, **_reduced_edge_data)

                self._reducing_map_dict[(u, v)] = (x_node, y_node)

            self._g_directed_reduced = NetworkClass(self._g_directed_reduced)

        return self._g_directed_reduced, self._reducing_map_dict

    @property
    def out_edges(self):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.out_edges

    def out_edges_by_node(self, node, data=False):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.out_edges(node, data=data)

    @property
    def in_edges(self):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.in_edges

    def in_edges_by_node(self, node, data=False):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.in_edges(node, data=data)

    def print_network(self):
        nx.draw(self.get_graph)
        plt.show()


if __name__ == "__main__":
    def get_base_graph():
        # init a triangle if we don't get a network graph
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edges_from([(0, 1, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                          (0, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 10}),
                          (1, 2, {EdgeConsts.WEIGHT_STR: 1, EdgeConsts.CAPACITY_STR: 15})])

        return g


    net = NetworkClass(get_base_graph())
    net.reducing_undirected2directed()
    adj = net.get_adjacency
    pairs = net.get_all_pairs()
