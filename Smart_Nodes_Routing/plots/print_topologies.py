from common.topologies import topology_zoo_loader
from common.network_class import NetworkClass
import numpy as np

if __name__ == "__main__":
    topologies_files = [
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/Claranet.txt', (3, 10, 12, 14)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/GoodNet.txt', (5, 7, 9, 12, 15)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/scale_free_30_nodes_47_seed.txt', (0, 1, 2, 3)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/GEANT.txt', (2, 4, 9, 23)),
        ('/home/idoye/PycharmProjects/Research_Implementing/common/graphs_gmls/China_Telecom.txt', (8, 18, 27, 28, 39))
    ]

    for topo in topologies_files:
        net = NetworkClass(topology_zoo_loader(topo[0]))
        net.print_network(hubs=topo[1])
        print("{}'s Links and capacities".format(net.get_title))
        num_links = net.get_num_edges
        num_cols = 4
        links_per_col = num_links / 4
        dict_of_links_capacities = net.get_edge_capacity_map()


        def link_capacity_generator():
            for link, capacity in dict_of_links_capacities.items():
                yield link, capacity


        remain = 0
        if int(links_per_col) < links_per_col:
            remain = (links_per_col - int(links_per_col)) * num_cols
            assert int(remain) == remain
            remain = int(remain)

        rows = list()
        capacity_generator = link_capacity_generator()

        for row_index in range(int(links_per_col)):
            row = list()
            for c in range(num_cols):
                row.append("{} & {}".format(*next(capacity_generator)))
            row[-1] += '\\\\'
            row = ' & '.join(row)
            rows.append(row)

        if remain > 0:
            row = list()
            for _ in range(remain):
                row.append("{} & {}".format(*next(capacity_generator)))

            row[-1] += '\\\\'
            row = ' & '.join(row)
            rows.append(row)

        rows = '\hline\hline\n' + '\n\hline\n'.join(rows) + '\n\hline\n'

        try:
            next(capacity_generator)
            raise Exception("Not all links printed")
        except StopIteration as e:
            pass

        print(rows)
