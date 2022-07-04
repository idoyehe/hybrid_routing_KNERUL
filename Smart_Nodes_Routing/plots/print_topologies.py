from common.topologies import topology_zoo_loader
from common.network_class import NetworkClass

if __name__ == "__main__":
    topologies_files = [
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/Claranet.txt', (3, 10, 12, 14)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/GoodNet.txt', (5, 7, 9, 12, 15)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/scale_free_30_nodes_47_seed.txt', (0, 1, 2, 3)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/GEANT.txt', (2, 4, 9, 23)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/China_Telecom.txt', (8, 18, 27, 28, 39))]

    for topo in topologies_files:
        net = NetworkClass(topology_zoo_loader(topo[0]))
        net.print_network(hubs=topo[1])
