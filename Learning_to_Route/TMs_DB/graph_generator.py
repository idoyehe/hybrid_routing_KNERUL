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


def parsing_learning_output(filepath: str):
    f = open(filepath, "rb")
    lines = f.readlines()
    f.close()
    lines = list(filter(lambda l: "ep_rew_mean" in str(l), lines))
    lines = list(map(lambda l: str(-1 * float(str(l).split("|")[2].split()[0])) + "\n", lines))
    f = open(filepath, "w")
    f.writelines(lines)
    f.close()


def plot_learning_convergence(filepath: str):
    f = open(filepath, "r")
    lines = f.readlines()
    lines = list(map(lambda l: float(str(l)), lines))
    f.close()
    fig = plt.figure(figsize=(20, 20))
    plt.plot(lines)
    plt.yticks(np.arange(1, 6.5, step=0.01))
    plt.show()
    plt.savefig("/home/idoye/PycharmProjects/Research_Implementing/Learning_to_Route/TMs_DB/{}.jpeg".format(
        filepath.split("/")[-1].split(".")[0]))


def static_routing_graphs():
    plot_graph(avg_cong_ratio=[1.79, 1.73, 1.7, 1.64, 1.196], title="Gravity 0.3",
               hlines=[(1.74, "Prev", 'r'), (1.49, "Avg K", 'k'), (1.16, "Oblivious", 'indigo')])
    plot_graph(avg_cong_ratio=[1.59, 1.47, 1.4, 1.35, 1.2], title="Gravity 0.6")
    plot_graph(avg_cong_ratio=[0.45, 0.41, 0.39, 0.36, 0.093], y_max=0.6, title="Standard Deviation Gravity 0.3")
    plot_graph(avg_cong_ratio=[1.26, 1.18, 1.14, 1.1, 1.2], title="Gravity 0.9")

    plot_graph(avg_cong_ratio=[1.2, 1.09, 1.07, 1.06, 1.09], y_max=1.4, title="Bimodal 1.0 Elephant 20%")
    plot_graph(avg_cong_ratio=[1.19, 1.09, 1.07, 1.06, 1.1], y_max=1.4, title="Bimodal 1.0 Elephant 40%",
               hlines=[(1.225, "Prev", 'r'), (1.18, "Avg K", 'k'), (1.125, "Oblivious", 'indigo')])
    plot_graph(avg_cong_ratio=[0.134, 0.078, 0.078, 0.068, 0.066], y_max=0.15,
               title="Standard Deviation Bimodal 1.0 Elephant 40%")
    plot_graph(avg_cong_ratio=[1.17, 1.09, 1.07, 1.06, 1.13], y_max=1.4, title="Bimodal 1.0 Elephant 60%")


if __name__ == "__main__":
    plot_learning_convergence(
        "/home/idoye/PycharmProjects/Research_Implementing/Learning_to_Route/TMs_DB/ConsoleOut_bimodal_10500.txt")
