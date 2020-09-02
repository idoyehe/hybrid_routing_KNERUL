import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def plot_baselines_graphs(title: str, avg_cong_ratio, error_bounds: list, hlines: list = None):
    assert len(avg_cong_ratio) == len(error_bounds)
    fig = plt.figure()
    bars = ['k=1 (Prev)', 'k=3', 'k=5', 'k=10', 'Oblivious']
    plt.errorbar(bars, avg_cong_ratio, yerr=error_bounds, fmt='ok', ecolor="green")
    y_max = max([x + y for x, y in zip(avg_cong_ratio, error_bounds)], )

    plt.xlabel("History Length / Routing Scheme")
    plt.ylabel("Average Congestion Ratio")
    if hlines:
        for hline in hlines:
            plt.axhline(y=hline[0], label=hline[1], color=hline[2], linestyle="dashed")
            y_max = max(y_max, hline[0])
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(1, y_max + 0.2, step=0.1))
    plt.show()


def plot_learning_convergence(filepath: str, right_limit: int, left_limit: int, y_ticks: np.array, title: str,
                              hlines: list = None):
    assert right_limit > left_limit
    f = open(filepath, "r")
    data = f.readlines()
    f.close()

    fig = plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

    data = list(map(lambda l: float(str(l)), data))[left_limit:right_limit]
    x = np.linspace(left_limit, right_limit, right_limit - left_limit)

    spline = make_interp_spline(x, data)
    data_new = spline(x)
    if hlines:
        for hline in hlines:
            plt.axhline(y=hline[0], label=hline[1], color=hline[2], linestyle="dashed")

    plt.plot(x, data_new)
    plt.yticks(y_ticks)
    plt.xlabel("# Learning Timesteps")
    plt.ylabel("Congestion Ratio")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig("{}_{}-{}.{}".format(filepath.split(".txt")[0] + "_plot", left_limit, right_limit, "jpeg"), bbox_inches='tight')
    plt.show()


def static_routing_graphs():
    plot_baselines_graphs(title="Gravity Sparsity 30%", avg_cong_ratio=[1.79, 1.73, 1.7, 1.64, 1.196],
                          error_bounds=[0.45, 0.41, 0.39, 0.36, 0.093],
                          hlines=[(1.75, "Prev", 'r'), (1.495, "Avg K", 'k'), (1.17, "Oblivious", 'indigo')])

    plot_baselines_graphs(title="Gravity Sparsity 60%", avg_cong_ratio=[1.59, 1.4673, 1.4073, 1.3532, 1.2],
                          error_bounds=[0.31, 0.2667, 0.234, 0.2004, 0.0633])

    plot_baselines_graphs(title="Gravity Sparsity 90%", avg_cong_ratio=[1.256, 1.175, 1.145, 1.109, 1.2],
                          error_bounds=[0.226, 0.129, 0.0995, 0.0745, 0.0633])

    plot_baselines_graphs(title="Bimodal Sparsity 100% Elephant 20%", avg_cong_ratio=[1.2, 1.097, 1.07484, 1.0609, 1.0902],
                          error_bounds=[0.14017, 0.0976, 0.084, 0.074144, 0.0685])

    plot_baselines_graphs(title="Bimodal Sparsity 100% Elephant 40%", avg_cong_ratio=[1.1984, 1.09750, 1.0752, 1.0603, 1.1],
                          error_bounds=[0.1340, 0.0925, 0.07849, 0.0681, 0.06598],
                          hlines=[(1.75, "Prev", 'r'), (1.48, "Avg K", 'k'), (1.17, "Oblivious", 'indigo')])

    plot_baselines_graphs(title="Bimodal Sparsity 100% Elephant 60%",
                          avg_cong_ratio=[1.1718, 1.08637, 1.06920, 1.0584091, 1.13667],
                          error_bounds=[0.11213, 0.0736, 0.06352, 0.056034, 0.0551])


def rl_routing_graphs():
    plot_learning_convergence(
        "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_gravity_350.txt",
        left_limit=0,
        right_limit=3000,
        y_ticks=np.arange(1, 7, step=0.1),
        title="Gravity 0.3 Sparsity, 350 matrices",
        hlines=[(1.79, "Prev", 'r'), (1.64, "Avg K", 'g'), (1.196, "Oblivious", 'indigo'), (1.17, "Convergence", 'k'), ])

    plot_learning_convergence(
        "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_gravity_350.txt",
        left_limit=500,
        right_limit=3500,
        y_ticks=np.arange(1, 2, step=0.05),
        title="Gravity 0.3 Sparsity, 350 matrices",
        hlines=[(1.79, "Prev", 'r'), (1.64, "Avg K", 'g'), (1.196, "Oblivious", 'indigo'), (1.17, "Convergence", 'k'), ])

    # plot_learning_convergence(
    #     "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_gravity_10500.txt", 0,
    #     4000,
    #     np.arange(1, 7, step=0.1),
    #     title="Gravity 0.3 Sparsity, 350 matrices",
    #     hlines=[(1.79, "Prev", 'r'), (1.64, "Avg K", 'k'), (1.196, "Oblivious", 'indigo'), (1.33, "Convergence", 'g'), ])
    #
    # plot_learning_convergence(
    #     "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_gravity_10500.txt", 500,
    #     3500,
    #     np.arange(1, 2, step=0.05),
    #     title="Gravity 0.3 Sparsity, 350 matrices",
    #     hlines=[(1.79, "Prev", 'r'), (1.64, "Avg K", 'k'), (1.196, "Oblivious", 'indigo'), (1.33, "Convergence", 'g'), ])

    pass

    # plot_learning_convergence(
    #     "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_bimodal_350.txt", 0,
    #     4000,
    #     np.arange(1, 11, step=0.2),
    #     title="Bimodal 1.0 Elephant 40%, 350 matrices",
    #     hlines=[(1.2, "Prev", 'r'), (1.09, "Avg K", 'k'), (1.1, "Oblivious", 'indigo'), (1.02, "Convergence", 'g'), ])

    # plot_learning_convergence(
    #     "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_bimodal_350.txt", 500,
    #     3500,
    #     np.arange(1, 1.2, step=0.01),
    #     title="Bimodal 1.0 Elephant 40%, 350 matrices",
    #     hlines=[(1.2, "Prev", 'r'), (1.09, "Avg K", 'k'), (1.1, "Oblivious", 'indigo'), (1.02, "Convergence", 'g'), ])
    #
    # plot_learning_convergence(
    #     "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_bimodal_10500.txt", 0,
    #     4000,
    #     np.arange(1, 11.2, step=0.2),
    #     title="Bimodal 1.0 Elephant 40%, 10500 matrices",
    #     hlines=[(1.2, "Prev", 'r'), (1.09, "Avg K", 'k'), (1.1, "Oblivious", 'indigo'), (1.04, "Convergence", 'g'), ])
    #
    # plot_learning_convergence(
    #     "C:\\Users\\IdoYe\\PycharmProjects\\Research_Implementing\\Learning_to_Route\\TMs_DB\\ConsoleOut_bimodal_10500.txt", 500,
    #     5000,
    #     np.arange(1, 1.2, step=0.01),
    #     title="Bimodal 1.0 Elephant 40%, 10500 matrices",
    #     hlines=[(1.2, "Prev", 'r'), (1.09, "Avg K", 'k'), (1.1, "Oblivious", 'indigo'), (1.04, "Convergence", 'g'), ])
    #


if __name__ == "__main__":
    rl_routing_graphs()
