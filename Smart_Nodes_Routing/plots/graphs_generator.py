import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_baselines_graphs(save_file, x_labels, y_labels, y_data, h_lines=None):
    y_max = 0
    title: str = "Hybrid-Routing MLU Vs. Optimal"
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(y_labels)):
        ax.plot(x_labels, y_data[i], label=y_labels[i])
        y_max = max(y_max, max(y_data[i]))

    plt.ylabel("Average Congestion Vs. Optimal Ratio")
    if h_lines is not None:
        for h_line in h_lines:
            ax.axhline(y=h_line[0], label=h_line[1], color=h_line[2], linestyle="dashed")
            y_max = max(y_max, h_line[0])
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(0, y_max + 1, step=1))
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    plt.savefig(save_file)
    plt.show()


if __name__ == "__main__":
    x_labels_4 = ['Link Weight Heuristic', 'First RL Phase', '1 Smart Node', '2 Smart Nodes', '3 Smart Nodes',
                  '4 Smart Nodes']
    x_labels_5 = ['Link Weight Heuristic', 'First RL Phase', '1 Smart Node', '2 Smart Nodes', '3 Smart Nodes',
                  '4 Smart Nodes', '5 Smart Nodes']
    y_labels = ["RL Training Set",
                "LP Training Set",
                "Test_0",
                "Test_1",
                "Test_2",
                "Test_3",
                "Test_4"
                ]
    h_lines = [[None, "Oblivious routing scheme", "red"], [None, "Mean TM routing scheme", "blue"]]

    # claranet gravity traffic plot
    h_lines[0][0] = 32.547
    h_lines[1][0] = 28.165
    y_data = [(16.959, 15.766, 12.986, 9.304, 6.883, 6.152),
              (16.955, 15.640, 12.306, 9.191, 6.351, 6.038),
              (17.147, 15.699, 11.903, 9.468, 7.164, 6.733),
              (16.295, 15.758, 12.435, 9.131, 7.476, 6.651),
              (15.926, 15.714, 11.937, 8.998, 6.910, 6.310),
              (16.724, 15.612, 12.316, 9.335, 6.849, 6.485),
              (16.321, 15.810, 12.791, 9.032, 7.089, 6.285)]
    save_file = 'claranet_gravity.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # goodnet gravity traffic plot
    h_lines[0][0] = 19.130
    h_lines[1][0] = 42.552
    y_data = [(18.295, 12.617, 11.321, 9.787, 9.149, 7.948, 6.519),
              (18.868, 12.506, 10.854, 9.117, 8.310, 5.860, 4.189),
              (17.965, 12.848, 11.476, 10.523, 9.952, 8.427, 6.712),
              (17.548, 12.897, 11.723, 10.290, 9.335, 8.352, 7.139),
              (18.307, 12.498, 11.326, 10.469, 9.517, 8.222, 7.327),
              (18.010, 12.731, 11.438, 10.116, 9.184, 8.138, 7.273),
              (18.010, 12.731, 11.438, 10.116, 9.184, 8.138, 7.273)]
    save_file = 'goodnet_gravity.png'
    plot_baselines_graphs(save_file, x_labels_5, y_labels, y_data, h_lines)

    # scale free 30 gravity traffic plot
    h_lines[0][0] = 26.675
    h_lines[1][0] = 34.784
    y_data = [(24.544, 14.920, 14.304, 12.545, 10.125, 9.173),
              (23.749, 13.855, 12.867, 10.499, 8.659, 6.276),
              (24.617, 15.377, 14.546, 12.494, 10.335, 8.999),
              (23.362, 15.680, 14.157, 12.185, 10.388, 8.903),
              (24.257, 15.198, 14.535, 12.491, 10.943, 8.816),
              (24.543, 15.561, 14.137, 12.544, 10.806, 9.067),
              (24.127, 15.008, 14.278, 12.057, 10.155, 8.794)]
    save_file = 'sf30_gravity.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # GEANT gravity traffic plot
    h_lines[0][0] = 48.530
    h_lines[1][0] = 67.114
    y_data = [(57.549, 14.100, 12.396, 10.434, 10.024, 9.813),
              (55.122, 13.178, 10.533, 8.373, 7.057, 6.860),
              (58.023, 13.732, 12.616, 10.432, 10.003, 9.680),
              (59.426, 13.711, 12.398, 10.629, 9.783, 9.403),
              (56.344, 14.190, 11.693, 10.550, 9.806, 9.337),
              (57.980, 13.991, 12.387, 10.772, 10.116, 9.841),
              (59.670, 14.359, 12.475, 10.732, 10.075, 9.758)]
    save_file = 'geant_gravity.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # China Telecom gravity traffic plot
    h_lines[0][0] = 52.701
    h_lines[1][0] = 61.051
    y_data = [(43.888, 25.574, 22.891, 19.217, 16.485, 14.665, 12.183),
              (43.856, 25.299, 22.574, 17.223, 13.234, 10.095, 8.326),
              (44.204, 24.490, 22.907, 18.628, 16.378, 14.441, 12.830),
              (43.610, 25.351, 22.664, 20.219, 16.930, 13.938, 12.549),
              (44.403, 25.916, 22.983, 19.601, 16.369, 14.127, 12.525),
              (45.009, 24.889, 22.792, 19.486, 16.552, 14.348, 12.623),
              (44.489, 25.207, 22.620, 19.490, 16.810, 14.137, 12.779)]
    save_file = 'china_telecom_gravity.png'
    plot_baselines_graphs(save_file, x_labels_5, y_labels, y_data, h_lines)

    # claranet bimodal traffic plot
    h_lines[0][0] = 14.134
    h_lines[1][0] = 20.256
    y_data = [(9.929, 8.081, 6.727, 5.743, 4.516, 4.327),
              (10.215, 7.454, 6.494, 5.856, 4.530, 4.313),
              (8.970, 7.754, 6.141, 5.319, 4.567, 4.234),
              (9.715, 8.414, 5.908, 5.687, 5.205, 4.325),
              (9.037, 7.981, 5.997, 4.862, 4.603, 4.214),
              (9.849, 7.706, 6.544, 5.560, 4.825, 4.181),
              (9.492, 7.410, 5.025, 5.360, 4.599, 4.451)]
    save_file = 'claranet_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # goodnet bimodal traffic plot
    h_lines[0][0] = 2.713
    h_lines[1][0] = 13.939
    y_data = [(3.315, 2.549, 2.496, 1.928, 1.917, 1.549, 0.984),
              (2.956, 2.675, 2.653, 2.093, 1.730, 1.394, 0.755),
              (3.043, 2.495, 2.087, 2.017, 1.628, 1.525, 1.074),
              (2.961, 2.543, 2.444, 2.162, 1.806, 1.159, 1.054),
              (2.779, 2.592, 1.986, 1.870, 1.240, 0.701, 0.432),
              (3.077, 2.590, 2.127, 1.961, 1.267, 1.081, 1.004),
              (3.262, 2.611, 2.468, 2.290, 1.901, 1.385, 1.056)]
    save_file = 'goodnet_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_5, y_labels, y_data, h_lines)

    # claranet custom bimodal traffic plot
    h_lines[0][0] = 16.368
    h_lines[1][0] = 24.303
    y_data = [(13.392, 11.105, 9.481, 8.303, 5.860, 5.426),
              (11.957, 10.476, 9.253, 8.346, 6.231, 5.388),
              (12.664, 11.320, 8.723, 6.751, 5.988, 4.752),
              (14.334, 13.643, 11.441, 9.460, 7.119, 6.823),
              (12.727, 11.653, 10.219, 8.278, 6.049, 5.684),
              (12.990, 11.360, 9.505, 8.736, 6.938, 6.299),
              (13.081, 12.103, 9.687, 7.807, 7.063, 6.541)]
    save_file = 'claranet_custom_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # goodnet custom bimodal traffic plot
    h_lines[0][0] = 4.488
    h_lines[1][0] = 19.378
    y_data = [(6.303, 5.826, 5.289, 4.178, 3.671, 2.630, 2.163),
              (6.841, 6.874, 5.300, 4.044, 3.084, 2.426, 1.887),
              (6.274, 5.822, 5.580, 4.240, 3.866, 3.522, 2.511),
              (5.863, 4.782, 4.537, 4.138, 3.709, 3.131, 2.118),
              (6.448, 5.943, 4.719, 4.023, 3.958, 3.752, 2.917),
              (6.916, 6.147, 5.187, 4.596, 3.813, 3.694, 2.815),
              (5.980, 5.126, 5.532, 4.202, 3.826, 3.560, 2.282)]
    save_file = 'goodnet_custom_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_5, y_labels, y_data, h_lines)

    # scale free 30 custom bimodal traffic plot
    h_lines[0][0] = 15.089
    h_lines[1][0] = 7.809
    y_data = [(3.481, 3.193, 3.128, 2.529, 2.236, 1.916),
              (3.329, 3.199, 2.539, 1.902, 1.362, 0.886),
              (3.559, 3.145, 2.822, 2.392, 2.189, 1.954),
              (3.245, 2.793, 2.660, 2.475, 2.250, 1.987),
              (3.396, 3.024, 2.849, 2.491, 2.376, 1.976),
              (3.432, 2.919, 2.675, 2.426, 2.361, 1.937),
              (3.135, 2.806, 2.626, 2.443, 2.084, 1.932)]
    save_file = 'sf30_custom_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # # GEANT custom bimodal traffic plot
    h_lines[0][0] = 1.169
    h_lines[1][0] = 28.524
    y_data = [(22.029, 16.185, 14.655, 7.297, 5.067, 0.077),
              (23.050, 16.169, 15.184, 6.174, 3.167, 0.002),
              (20.813, 17.540, 13.116, 7.326, 3.601, 0.045),
              (21.925, 17.530, 15.241, 7.553, 4.833, 0.041),
              (21.797, 16.921, 14.781, 8.233, 4.045, 0.096),
              (21.386, 17.123, 15.202, 7.622, 5.006, 0.033),
              (22.158, 19.081, 15.179, 7.784, 4.588, 0.059)]
    save_file = 'geant_custom_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_4, y_labels, y_data, h_lines)

    # China Telecom custom bimodal traffic plot
    h_lines[0][0] = 0.047
    h_lines[1][0] = 12.725
    y_data = [(1.240, 0.723, 0.535, 0.106, 0.0099, 0.194, 0.553),
              (1.417, 0.672, 0.223, 0.142, 0.1303, 0.097, 0.010),
              (1.343, 0.628, 0.229, 0.053, 0.0066, 0.493, 0.604),
              (1.136, 0.373, 0.177, 0.035, 0.6106, 0.432, 0.557),
              (1.439, 0.682, 0.292, 0.057, 0.3127, 0.396, 0.229),
              (1.358, 0.496, 0.325, 0.094, 0.0932, 0.153, 0.743),
              (1.205, 0.610, 0.511, 0.070, 0.3433, 0.047, 0.270)]
    save_file = 'china_telecom_custom_bimodal.png'
    plot_baselines_graphs(save_file, x_labels_5, y_labels, y_data, h_lines)
