"""
Created on 5 Jul 2017
@author: asafvaladarsky

refactoring on 04/04/2020
@by: Ido Yehezkel
"""

import numpy as np


def build_edges_map(graph_adjacency):
    """
    @param graph_adjacency: symmetric matrix of 0,1; cell[i,j] holds 1 iff i and j are adjacent nodes
    @return: num of edges, map for each node its ingoing and outgoing edges
    """
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
