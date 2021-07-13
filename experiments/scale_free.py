import networkx as nx
import matplotlib.pyplot as plt
from common.network_class import *
if __name__ == "__main__":
    n=28
    s = 35
    G = nx.generators.scale_free_graph(n,seed=s) #,alpha=0.9,beta=0.05,gamma=0.05))
    for edge in G.edges:
        G.edges[edge][EdgeConsts.CAPACITY_STR] = 10000000000
    G = nx.Graph(G)
    net = NetworkClass(G)
    print(net.get_num_edges)
    net.print_network()
    nx.write_gml(G,"C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\common\\graphs_gmls\\scale_free_{}_nodes_{}_seed.txt".format(n,s))
