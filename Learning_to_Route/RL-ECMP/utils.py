import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_path_length_and_edges(g, path, weight=None):
    path_len = 0
    path_edges = []
    for i in range(len(path) - 1):
        path_len += 1 if not weight else g[path[i]][path[i + 1]][weight]
        path_edges.append((path[i], path[i + 1]))
    return path_len, path_edges
