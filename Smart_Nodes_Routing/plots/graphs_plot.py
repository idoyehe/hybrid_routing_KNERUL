import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sys import argv
from argparse import ArgumentParser
from parsing_data_results import parsing_data_results, parse_rl_optimization_cross_topologies


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses TMs Generating script arguments")
    parser.add_argument("-topo", "--topology_name", type=str, help="The topology name")
    parser.add_argument("-traffic", "--traffic_name", type=str, help="The traffic distribution name")
    parser.add_argument("-obliv_base", "--oblivious_baseline", type=eval, help="If oblivious is the baseline",
                        default=False)
    parser.add_argument("-pgf", "--pgf", type=eval, help="PGF format", default=False)
    options = parser.parse_args(args)
    return options


def plot_baselines_graphs(save_file, x_labels, y_data, oblivious_baseline, h_lines=None):
    bar_design = dict()
    bar_design["Non-Key Nodes Train Set"] = {"hatch": None}
    bar_design["Key Nodes Train Set"] = {"hatch": '\\'}
    bar_design["Test Sets"] = {"hatch": "."}

    fontsize = 11
    fig = plt.figure(figsize=(7.3, 5))
    ax = plt.subplot()
    ax.autoscale(enable=True)
    ind = np.arange(len(x_labels))

    offset = -1
    offset_value = 0.28
    width = 0.2
    y_tick_offset = np.inf
    y_max = 0
    for y_label, value in y_data.items():
        y_data, yerr = tuple(zip(*value))
        ax.bar(ind + (offset * offset_value), y_data, width=width, align='center', label=y_label, yerr=yerr,
               error_kw=dict(lw=0.8, capsize=1.8, capthick=0.8, ecolor='black'),
               fill=False,
               hatch=bar_design[y_label]["hatch"],
               )
        offset += 1
        y_tick_offset = min(y_tick_offset, y_data[-2] - y_data[-1])
        y_max = max(y_max, max(y_data))

    plt.xticks(ind, x_labels, rotation=0, fontsize=fontsize)

    plt.ylabel("Maximum Link Utilization Ratio", fontsize=fontsize)
    for h_line in h_lines:
        ax.axhline(y=h_line[3], label=h_line[0], color=h_line[1], linestyle=h_line[2])
        y_max = max(y_max, h_line[3])

    plt.legend(fontsize=fontsize)
    ax.yaxis.grid(alpha=0.5)  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    bottom = 0.0 if oblivious_baseline else 1.0
    plt.ylim(bottom=bottom)
    plt.yticks(np.arange(bottom, 1.34, step=y_tick_offset * 5.5), fontsize=fontsize)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()


def plot_cross_topologies_graphs(save_file, topologies, y_data):
    bar_design = dict()
    bar_design["Uniform Link Weight\nInitialized RL"] = {"color": "#aac2a1", "alpha": 1.0}
    bar_design["Optimized Link Weight\nInitialization"] = {"color": "#cef0cc", "alpha": 1.0}
    bar_design["Optimized Link Weight\nInitialized RL"] = {"color": "darkgreen", "alpha": 1.0}
    fontsize = 11
    fig = plt.figure(figsize=(7.3, 5))
    ax = plt.subplot()
    ax.autoscale(enable=True)
    ind = np.arange(len(topologies))

    offset = -1
    offset_value = 0.15
    width = 0.15
    y_tick_offset = np.inf
    y_max = 0
    for y_label, value in y_data.items():
        y_data, yerr = tuple(zip(*value))
        ax.bar(ind + (offset * offset_value), y_data, width=width, align='center', label=y_label, yerr=yerr,
               error_kw=dict(ecolor='black', capsize=5),
               color=bar_design[y_label]["color"], alpha=bar_design[y_label]["alpha"])
        offset += 1
        y_tick_offset = min(y_tick_offset, np.abs(y_data[-2] - y_data[-1]))
        y_max = max(y_max, max(y_data))

    plt.xticks(ind, topologies, rotation=0, fontsize=fontsize)

    plt.ylabel("Maximum Link Utilization Ratio", fontsize=fontsize)

    plt.legend(fontsize=fontsize)
    ax.yaxis.grid()  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest
    plt.ylim(bottom=1.0)
    plt.yticks(np.arange(1.0, 1.35, step=y_tick_offset * 0.15), fontsize=fontsize)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()


if __name__ == "__main__":
    args = _getOptions()
    topology_name = args.topology_name
    traffic_name = args.traffic_name
    oblivious_baseline = args.oblivious_baseline
    pgf = args.pgf
    suffix = "png"

    if pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        suffix = "pgf"

    baseline_name = "oblivious" if oblivious_baseline else "optimal"

    save_file = "{}_{}_baseline_{}.{}".format(topology_name, traffic_name, baseline_name, suffix)

    x_labels, y_data, h_lines = parsing_data_results(topology_name, traffic_name, oblivious_baseline)
    plot_baselines_graphs(save_file, x_labels, y_data, oblivious_baseline, h_lines)

    # save_file = "{}_{}.{}".format("link_weights_init", traffic_name,suffix)

    # topologies, y_data = parse_rl_optimization_cross_topologies(traffic_name)
    # plot_cross_topologies_graphs(save_file, topologies, y_data)
