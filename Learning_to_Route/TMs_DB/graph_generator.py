import matplotlib.pyplot as plt
import numpy as np


def plot_graph(avg_cong_ratio, title: str = ""):
    fig = plt.figure()
    bars = ['k=1 (Prev)', 'k=3', 'k=5', 'k=10', 'Oblivious']
    plt.bar(bars, avg_cong_ratio)
    plt.xlabel("Routing Scheme")
    plt.ylabel("Average Congestion Ratio")
    plt.title(title)
    plt.yticks(np.arange(0, 1.7, step=0.1))
    plt.show()


if __name__ == "__main__":
    plot_graph(avg_cong_ratio=[1.61, 1.58, 1.56, 1.52, 1.31], title="Gravity 0.3")
    plot_graph(avg_cong_ratio=[1.48, 1.38, 1.35, 1.29, 1.37], title="Gravity 0.6")
    plot_graph(avg_cong_ratio=[1.22, 1.16, 1.13, 1.1, 1.41], title="Gravity 0.9")

    plot_graph(avg_cong_ratio=[1.18, 1.09, 1.07, 1.05, 1.26], title="Bimodal 1.0 Elephant 20%")
    plot_graph(avg_cong_ratio=[1.18, 1.09, 1.07, 1.05, 1.27], title="Bimodal 1.0 Elephant 40%")
    plot_graph(avg_cong_ratio=[1.16, 1.08, 1.07, 1.05, 1.32], title="Bimodal 1.0 Elephant 60%")