"""
Created on 10 Mar 2017
@author: asafvaladarsky

refactoring on 24/04/2020
@by: Ido Yehezkel
"""
from common.consts import EdgeConsts
import networkx as nx
from common.logger import logger
import matplotlib.pyplot as plt
from common.utils import *
from random import shuffle
import pickle
from collections import defaultdict


class NetworkClass(object):

    def __init__(self, topo):
        logger.info("Creating Network Class")

        self._graph = topo.to_directed().copy()
        assert nx.is_directed(self.get_graph)
        self._is_directed = nx.is_directed(self.get_graph)

        self._all_edges = self._graph.edges

        self._capacities = None  # this is a vector of capacities
        self._edge_capacity_map = None  # this is a dict of capacities
        self._all_pairs = None
        self._num_nodes = len(self._graph.nodes)
        self._num_edges = len(self._all_edges)
        self._actual_weight = np.zeros(self._graph.number_of_edges())
        self._set_adjacency()  # mark all adjacent nodes
        self._reducing_map_dict = None
        self._id2edge_map = None
        self._edge2id_map = None
        self._outgoing_capacity = None
        self._ingoing_capacity = None
        self._total_capacity = 0
        self._flows = None
        self._title = None
        self._chosen_pairs = None
        self._elephant_percentages = None
        self._smart_nodes: tuple = tuple()
        self._smart_nodes_spr: dict = dict()
        print("Network {} has been created".format(self.get_title))

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
    def get_title(self):
        if (not hasattr(self, "_title") or self._title is None) and "Name" in self.get_graph.graph:
            self._title = self.get_graph.graph["Name"]
        return self._title

    def set_title(self, title):
        self._title = title

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

    @property
    def get_smart_nodes(self):
        return self._smart_nodes

    def set_smart_nodes(self, smart_nodes: tuple):
        for node in smart_nodes:
            assert node in self.nodes
        self._smart_nodes = smart_nodes

    @property
    def get_smart_nodes_spr(self):
        return self._smart_nodes_spr

    def set__smart_nodes_spr(self, _smart_nodes_spr):
        self._smart_nodes_spr = _smart_nodes_spr

    def get_all_pairs(self):
        """ return all nodes pairs"""
        if self._all_pairs is None:
            logger.debug("build all node pair list")
            self._all_pairs = []
            for src_ind in self._graph.nodes:
                for dst_ind in self._graph.nodes:
                    if dst_ind == src_ind or not nx.has_path(self.get_graph, src_ind, dst_ind):
                        continue
                    self._all_pairs.append((src_ind, dst_ind))
                    assert dst_ind != src_ind and nx.has_path(self.get_graph, src_ind, dst_ind)
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
        num_edges = self.get_num_edges
        ingoing = np.zeros((len(graph_adjacency), num_edges))
        outgoing = np.zeros((len(graph_adjacency), num_edges))
        self._capacities = np.zeros(num_edges)
        self._id2edge_map = dict()
        self._edge2id_map = dict()
        self._edge_capacity_map = dict()
        edge_id = 0
        for i in range(len(graph_adjacency)):
            for j in range(len(graph_adjacency)):
                if graph_adjacency[i][j] == 1:
                    outgoing[i, edge_id] = 1
                    ingoing[j, edge_id] = 1
                    self._capacities[edge_id] = self.get_edge_key((i, j), EdgeConsts.CAPACITY_STR)
                    self._id2edge_map[edge_id] = (i, j)
                    self._edge2id_map[(i, j)] = edge_id
                    self._edge_capacity_map[(i, j)] = self._capacities[edge_id]
                    edge_id += 1
        return ingoing, outgoing, self._capacities

    def get_edge_capacity_map(self):
        if self._edge_capacity_map is None:  # for happens only once
            self.build_edges_map()
        return self._edge_capacity_map

    def get_id2edge_map(self):
        if self._id2edge_map is None:
            self.build_edges_map()
        return self._id2edge_map

    def get_edge2id_map(self):
        if self._edge2id_map is None:
            self.build_edges_map()
        return self._edge2id_map

    def get_id2edge(self, id):
        return self.get_id2edge_map()[id]

    def get_edge2id(self, src, dst):
        return self.get_edge2id_map()[src, dst]

    @property
    def out_edges(self):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.out_edges

    def out_edges_by_node(self, node, data=False):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.out_edges(node, data=data)

    def all_simple_paths(self, source, target):
        return nx.all_simple_paths(self.get_graph, source, target)

    @property
    def in_edges(self):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.in_edges

    def in_edges_by_node(self, node, data=False):
        assert isinstance(self.get_graph, nx.DiGraph)
        return self.get_graph.in_edges(node, data=data)

    def print_network(self, label=None, hubs=None):
        g = self.get_graph
        pos = nx.spring_layout(g)
        if label is not None:
            edge_labels = dict([((u, v,), d[label]) for u, v, d in g.edges(data=True)])
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        values = ['none' for node in g.nodes()]
        if hubs is not None:
            values = ['gray' for node in g.nodes()]
            for hub in hubs:
                values[hub] = 'forestgreen'

        nx.draw(g, pos, with_labels=True, node_shape="o", node_color=values, edgecolors='black')
        plt.show()

    def __capacity_map(self):
        if self._outgoing_capacity is None or self._ingoing_capacity is None:
            assert self.g_is_directed
            self._outgoing_capacity = defaultdict(int)
            self._ingoing_capacity = defaultdict(int)
            self._total_capacity = 0
            for u, v in self.edges:
                edge_capacity = self.get_edge_key((u, v), key=EdgeConsts.CAPACITY_STR)
                self._ingoing_capacity[v] += edge_capacity
                self._outgoing_capacity[u] += edge_capacity
                self._total_capacity += edge_capacity
        return self._outgoing_capacity, self._ingoing_capacity, self._total_capacity

    def gravity_traffic_map(self, scale=1.0):
        if self._flows is None:
            self.__capacity_map()
            self._flows = []

            for src, dst in self.get_all_pairs():
                flow_size = np.round(self._outgoing_capacity[src] * self._ingoing_capacity[dst] / self._total_capacity, 4)
                self._flows.append((src, dst, scale * flow_size))
        return self._flows

    def elephant_percentages(self):
        if self._elephant_percentages is None:
            self.__capacity_map()
            self._elephant_percentages = np.zeros(shape=(self.get_num_nodes,), dtype=np.float64)
            for u in self.nodes:
                self._elephant_percentages[u] = self._outgoing_capacity[u] / self._total_capacity

            assert error_bound(np.sum(self._elephant_percentages), 1.0)
        return self._elephant_percentages

    def __randomize_pairs(self, percent):
        all_pairs_copy = list(self.get_all_pairs())
        # shuffle the pairs
        shuffle(all_pairs_copy)
        num_pairs_selected = int(np.ceil(len(all_pairs_copy) * percent))
        chosen_pairs = list()
        while len(chosen_pairs) != num_pairs_selected:
            pair_index = np.random.choice(len(all_pairs_copy))
            chosen_pairs.append(all_pairs_copy[pair_index])
            all_pairs_copy.pop(pair_index)
        return chosen_pairs

    def choosing_pairs(self, percent, static=False):
        if self._chosen_pairs is None or static is False:
            self._chosen_pairs = self.__randomize_pairs(percent)
        return self._chosen_pairs

    def get_node_degree(self, node_id):
        return self.get_graph.degree[node_id]

    def get_sorted_degrees(self):
        return sorted(self.get_graph.degree, key=lambda x: x[1], reverse=True)

    def store_network_object(self, file_path, env_train_observation):
        self.env_train_observation = env_train_observation
        file_path += "{}_object.pkl".format(self.get_title)
        output = open(file_path, 'wb')
        pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    @staticmethod
    def load_network_object(file_path):
        input = open(file_path, 'rb')
        net = pickle.load(input)
        input.close()
        return net


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
    adj = net.get_adjacency
    pairs = net.get_all_pairs()
