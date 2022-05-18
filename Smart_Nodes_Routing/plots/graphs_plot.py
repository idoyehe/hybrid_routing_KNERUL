import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sys import argv
from argparse import ArgumentParser
from parsing_data_results import parsing_data_results, parse_rl_optimization_cross_topologies


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-topo", "--topology_name", type=str, help="The topology name")
    parser.add_argument("-traffic", "--traffic_name", type=str, help="The traffic distribution name")
    parser.add_argument("-obliv_base", "--oblivious_baseline", type=eval, help="If oblivious is the baseline", default=False)
    options = parser.parse_args(args)
    return options


def plot_baselines_graphs(save_file, x_labels, y_data, oblivious_baseline, h_lines=None):
    fig = plt.figure(figsize=(14, 10), dpi=80)
    ax = plt.subplot(111)
    ax.autoscale(enable=True)
    ind = np.arange(len(x_labels))

    offset = -1
    offset_value = 0.28
    width = 0.2
    y_tick_offset = np.inf
    y_max = 0
    for y_label, value in y_data.items():
        y_data, yerr = tuple(zip(*value))
        ax.bar(ind + (offset * offset_value), y_data, width=width, align='center', label=y_label, yerr=yerr, ecolor='black', capsize=5)
        offset += 1
        y_tick_offset = min(y_tick_offset, y_data[-2] - y_data[-1])
        y_max = max(y_max, max(y_data))

    plt.xticks(ind, x_labels, rotation=45, fontsize=14)

    plt.ylabel("Maximum Link Utilization Ratio", fontsize=14)
    for h_line in h_lines:
        ax.axhline(y=h_line[2], label=h_line[0], color=h_line[1], linestyle="dashed")
        y_max = max(y_max, h_line[2])

    plt.legend()
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    bottom = 0.0 if oblivious_baseline else 1.0
    plt.ylim(bottom=bottom)
    # plt.yticks(np.arange(bottom, y_max , step=y_tick_offset * 0.0001), fontsize=14)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()


def plot_cross_topologies_graphs(save_file, x_labels, topo_data):
    fig = plt.figure(figsize=(14, 10), dpi=80)
    ax = plt.subplot(111)
    ax.autoscale(enable=True)
    ind = np.arange(len(x_labels))

    offset = -2
    offset_value = 0.15
    width = 0.15
    y_tick_offset = np.inf
    y_max = 0
    for y_label, value in topo_data.items():
        y_data, yerr = tuple(zip(*value))
        ax.bar(ind + (offset * offset_value), y_data, width=width, align='center', label=y_label, yerr=yerr, ecolor='black', capsize=5)
        offset += 1
        y_tick_offset = min(y_tick_offset, y_data[-2] - y_data[-1])
        y_max = max(y_max, max(y_data))

    plt.xticks(ind, x_labels, rotation=45, fontsize=14)

    plt.ylabel("Maximum Link Utilization Ratio", fontsize=14)

    plt.legend()
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.ylim(bottom=1.0)
    plt.yticks(np.arange(1.0, 1.35, step=y_tick_offset * 3.5), fontsize=14)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()


if __name__ == "__main__":
    args = _getOptions()
    topology_name = args.topology_name
    traffic_name = args.traffic_name
    oblivious_baseline = args.oblivious_baseline

    # baseline_name = "oblivious" if oblivious_baseline else "optimal"
    #
    # save_file = "{}_{}_baseline_{}".format(topology_name, traffic_name, baseline_name)
    #
    # x_labels, y_data, h_lines = parsing_data_results(topology_name, traffic_name, oblivious_baseline)
    # plot_baselines_graphs(save_file, x_labels, y_data, oblivious_baseline, h_lines)

    save_file = "{}_{}".format("link_weights_init", traffic_name)
    x_labels, topo_data = parse_rl_optimization_cross_topologies(traffic_name)
    plot_cross_topologies_graphs(save_file, x_labels, topo_data)
