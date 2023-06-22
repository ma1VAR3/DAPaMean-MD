import pickle
import os
import json
import numpy as np
from dash import Dash, html, dcc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def get_figure(
    x_axis_data,
    x_label,
    y_axis_data,
    y_label,
    title,
    legend,
    y_axis_tick=0.5,
    legend_prefix="",
):
    fig = go.Figure()
    maks = [
        "hourglass",
        "circle",
        "square",
        "star",
        "square-open",
        "x",
        "bowtie",
        "circle-open",
        "x-open",
    ]
    mp_maks = ["o", "^", "s", "*", "x", "P"]
    line_styles = [
        "-",
        "-.",
        "--",
        ":",
        "-",
        "--",
        ":",
        "-.",
        "-",
        "--",
        ":",
        "-.",
        "-",
        "--",
        ":",
        "-.",
    ]

    fsize = 15
    tsize = 14

    tdir = "in"

    major = 5.0
    minor = 3.0

    style = "default"

    plt.style.use(style)
    plt.rcParams["font.size"] = fsize
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.direction"] = tdir
    plt.rcParams["ytick.direction"] = tdir
    plt.rcParams["xtick.major.size"] = major
    plt.rcParams["xtick.minor.size"] = minor
    plt.rcParams["ytick.major.size"] = major
    plt.rcParams["ytick.minor.size"] = minor
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(y_axis_tick))
    # xsize = 8
    # ysize = 5
    # plt.figure(figsize=(xsize, ysize))
    plt.title(title, fontsize=tsize)

    for i in range(len(y_axis_data)):
        fig.add_trace(
            go.Scatter(
                x=x_axis_data,
                y=y_axis_data[i],
                mode="lines+markers",
                name=legend_prefix + str(legend[i]),
                marker_symbol=maks[i],
            )
        )
        plt.plot(
            x_axis_data,
            y_axis_data[i],
            label=legend_prefix + str(legend[i]),
            marker=mp_maks[i],
            linestyle=line_styles[i],
        )
    plt.legend()
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x_label, labelpad=5, fontsize=12)
    plt.ylabel(y_label, labelpad=5, fontsize=12)
    plt.show()
    fig.update_layout(
        # scene = dict(
        #     xaxis_title=x_label,
        #     yaxis_title=y_label,
        # ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        title_text=title,
        title_x=0.5,
    )

    return fig


config = None
with open("./config.json", "r") as jsonfile:
    config = json.load(jsonfile)
    print("Configurations loaded from config.json")
    jsonfile.close()

file_path_base = "./results syn num samp scaling 4 eps/{}/{}/{}/losses.npy"
conc_algos = ["baseline", "coarse_mean", "quantiles"]
epsilons = config["epsilons"]
factor1 = 63
factor2 = 40

losses_base_rmse = []
losses_base2_rmse = []
losses_cm_rmse = []
losses_cm_bf_rmse = []
losses_q_wrap_rmse = []
losses_q_best_rmse = []

losses_base_worst = []
losses_base2_worst = []
losses_cm_worst = []
losses_cm_bf_worst = []
losses_q_wrap_worst = []
losses_q_best_worst = []

for e in epsilons:
    losses_base_e = np.load(file_path_base.format("baseline", "wrap", str(e)))
    losses_base2_e = np.load(file_path_base.format("baseline2", "best_fit", str(e)))
    losses_cm_e = np.load(file_path_base.format("coarse_mean", "wrap", str(e)))
    losses_cm_bf_e = np.load(file_path_base.format("coarse_mean", "best_fit", str(e)))
    losses_q_wrap_e = np.load(file_path_base.format("quantiles", "wrap", str(e)))
    losses_q_best_e = np.load(file_path_base.format("quantiles", "best_fit", str(e)))

    loaded_set = [
        losses_base_e,
        losses_base2_e,
        losses_cm_e,
        losses_cm_bf_e,
        losses_q_wrap_e,
        losses_q_best_e,
    ]
    store_set_rmse = [
        losses_base_rmse,
        losses_base2_rmse,
        losses_cm_rmse,
        losses_cm_bf_rmse,
        losses_q_wrap_rmse,
        losses_q_best_rmse,
    ]
    store_set_worst = [
        losses_base_worst,
        losses_base2_worst,
        losses_cm_worst,
        losses_cm_bf_worst,
        losses_q_wrap_worst,
        losses_q_best_worst,
    ]

    for loaded_loss, store_array in zip(loaded_set, store_set_rmse):
        dim_rmses = np.sqrt(np.mean(loaded_loss**2, axis=1))
        overall_rmse = np.sqrt(np.mean(dim_rmses**2))
        store_array.append(overall_rmse)

    for loaded_loss, store_array in zip(loaded_set, store_set_worst):
        exp_perc = np.percentile(loaded_loss, 100, axis=0)
        mean_perc = np.mean(exp_perc)
        store_array.append(mean_perc)

print(losses_base_rmse)
print(losses_cm_rmse)
print(losses_cm_bf_rmse)
print(losses_q_wrap_rmse)
print(losses_q_best_rmse)

fig = get_figure(
    epsilons,
    "Epsilon per dimension",
    [
        losses_base_rmse,
        losses_base2_rmse,
        losses_cm_rmse,
        losses_cm_bf_rmse,
        losses_q_wrap_rmse,
        losses_q_best_rmse,
    ],
    "Error",
    "Error vs Epsilon per Dimension",
    [
        "Baseline",
        "AAA + best",
        "Levy + wrap",
        "Levy + best",
        "DAPaMean-MD wrap",
        "DAPaMean-MD best",
    ],
    legend_prefix="",
)

fig.show()

fig = get_figure(
    epsilons,
    "Epsilon per dimension",
    [
        losses_base_worst,
        losses_base2_worst,
        losses_cm_worst,
        losses_cm_bf_worst,
        losses_q_wrap_worst,
        losses_q_best_worst,
    ],
    "Error",
    "Error vs Epsilon per Dimension",
    [
        "Baseline",
        "AAA + best",
        "Levy + wrap",
        "Levy + best",
        "DAPaMean-MD wrap",
        "DAPaMean-MD best",
    ],
    legend_prefix="",
)

fig.show()

# mul_q = 40
# mul_b = 63

# budget_base = eps * mul_b
# budget_q = eps * mul_q

# fig = get_figure(
#     eps,
#     "Epsilon per dimension",
#     [budget_base, budget_q],
#     "Cumulative Epsilon",
#     "Cumulative Epsilon vs Epsilon per Dimension",
#     ["Baseline/Levy", "DAPaMean-MD"],
#     y_axis_tick=2,
#     legend_prefix="",
# )

# fig.show()
