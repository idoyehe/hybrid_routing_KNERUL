from Learning_to_Route.common.size_consts import SizeConsts
from Learning_to_Route.common.consts import Consts
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class NetworkGraph:

    def __init__(self, num_nodes: int, avg_degree: int, min_capacity: float = 0.1, max_capacity: float = 10.0, seed: int = 0):
        np.random.seed(seed=seed)
        self.inner_graph = nx.DiGraph()
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.__pairs__ = list()
        self.__build_graph_edges__()

    def number_of_nodes(self):
        return self.num_nodes

    def nodes(self):
        return list(self.inner_graph.nodes)

    def __generating_outgoing_edges_degrees__(self) -> list:
        while True:
            temp = [np.random.randint(1, self.num_nodes - 1) for _ in range(self.num_nodes)]
            if int(np.mean(temp)) == self.avg_degree:
                return temp

    def __single_vertex_outgoing_edges__(self, current_vertex, outgoing_degree) -> list:
        marked = [False for _ in range(self.num_nodes)]
        marked[current_vertex] = True
        outgoing_edges_list = list()
        for _ in range(outgoing_degree):
            neighbor = None
            while neighbor is None:
                neighbor = np.random.randint(0, self.num_nodes)
                if not marked[neighbor]:
                    marked[neighbor] = True
                else:
                    neighbor = None
            self.__pairs__.append((current_vertex, neighbor))
            cap = np.random.uniform(low=self.min_capacity, high=self.max_capacity) * SizeConsts.ONE_Gb
            outgoing_edges_list.append((current_vertex, neighbor, cap))
        return outgoing_edges_list

    def pairs(self) -> list:
        return self.__pairs__

    def __build_graph_edges__(self):
        self.inner_graph.add_nodes_from(range(self.num_nodes))
        outgoing_degree_per_vertex = self.__generating_outgoing_edges_degrees__()
        all_edges_list = list()
        for current_vertex in range(self.num_nodes):
            outgoing_degree = outgoing_degree_per_vertex[current_vertex]
            all_edges_list.extend(self.__single_vertex_outgoing_edges__(current_vertex=current_vertex, outgoing_degree=outgoing_degree))

        self.inner_graph.add_weighted_edges_from(all_edges_list, weight=Consts.CAPACITY_STR)

    def draw_network_graph(self):
        nx.draw(self.inner_graph, with_labels=True, font_weight='bold')
        plt.show()

    def __getitem__(self, item):
        return self.inner_graph[item]


# g = NetworkGraph(9, 5)
# g.draw_network_graph()
# print(g[0])
# print(generate_tm(g, 0.2, Consts.GRAVITY))
