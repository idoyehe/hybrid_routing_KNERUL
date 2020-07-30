import matplotlib.pyplot as plt
import numpy as np


def plot_graph(avg_cong_ratio, title: str = "", y_max=1.9, hlines: list = None):
    fig = plt.figure()
    bars = ['k=1 (Prev)', 'k=3', 'k=5', 'k=10', 'Oblivious']
    plt.bar(bars, avg_cong_ratio)
    plt.xlabel("Routing Scheme")
    plt.ylabel("Average Congestion Ratio")
    if hlines:
        for hline in hlines:
            plt.axhline(y=hline[0], label=hline[1], color=hline[2], linestyle="dashed")
    plt.title(title)
    plt.legend()
    plt.yticks(np.arange(0, y_max, step=0.1))
    plt.show()


if __name__ == "__main__":
    plot_graph(avg_cong_ratio=[1.61, 1.58, 1.56, 1.52, 1.31], title="Gravity 0.3",
               hlines=[(1.73, "Prev", 'r'), (1.48, "Avg K", 'k'), (1.16, "Oblivious", 'indigo')])
    plot_graph(avg_cong_ratio=[1.48, 1.38, 1.35, 1.29, 1.37], title="Gravity 0.6")
    plot_graph(avg_cong_ratio=[1.22, 1.16, 1.13, 1.1, 1.41], title="Gravity 0.9")

    plot_graph(avg_cong_ratio=[1.18, 1.09, 1.07, 1.05, 1.26], y_max=1.4, title="Bimodal 1.0 Elephant 20%")
    plot_graph(avg_cong_ratio=[1.18, 1.09, 1.07, 1.05, 1.27], y_max=1.4, title="Bimodal 1.0 Elephant 40%",
               hlines=[(1.225, "Prev", 'r'), (1.18, "Avg K", 'k'), (1.125, "Oblivious", 'indigo')])
    plot_graph(avg_cong_ratio=[1.16, 1.08, 1.07, 1.05, 1.32], y_max=1.4, title="Bimodal 1.0 Elephant 60%")
