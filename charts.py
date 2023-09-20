import pickle
import os
import json
import numpy as np
from dash import Dash, html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook"
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def get_subplots_nonSynthetic(
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
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        # shared_yaxes=True,
                        horizontal_spacing=0.05, 
                        vertical_spacing=0.06, 
                        subplot_titles=("Synthetic Data with User Scaling",
                                        ""
                                        )
                        )

#for reference only, DO NOT UNCOMMENT
# [
#  0       losses_base2_rmse,
#  1       losses_base2_bf_rmse,
#  2       losses_cm_rmse,
#  3       losses_cm_bf_rmse,
#  4       losses_q_wrap_rmse,
#  5       losses_q_best_rmse,
#  6       losses_base2_worst,
#  7       losses_base2_bf_worst,
#  8       losses_cm_worst,
#  9      losses_cm_bf_worst,
#  10      losses_q_wrap_worst,
#  11      losses_q_best_worst,
#     ]

    #creating an array of y values 
    y = []
    for i in y_axis_data:
        y.append(i)
    # print(y)
    # print(x_axis_data)

    # Add traces to the first subplot (RMSE)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[1], 
                             mode='lines+markers', 
                             name='Array-Averaging',
                             marker_color = 'green',
                             marker_symbol=maks[0]),
                row=1, col=1)  
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[3], 
                             mode='lines+markers', 
                             name='Levy Algorithm-based Clipping',
                             marker_color = 'blue', 
                             marker_symbol=maks[1]),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[5], 
                             mode='lines+markers', 
                             name='Quantile-based Clipping',
                             marker_color = 'orange', 
                             marker_symbol=maks[2]),
                row=1, col=1)

    # Add a trace to the second subplot (MAE)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[7], 
                             mode='lines+markers', 
                             name='Array-Averaging',
                             marker_color = 'green',
                             showlegend=False,
                             marker_symbol=maks[0]),
                row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[9], 
                             mode='lines+markers', 
                             name='Levy Algorithm-based Clipping',
                             marker_color = 'blue',
                             showlegend=False,
                             marker_symbol=maks[1]),
                row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[11], 
                             mode='lines+markers', 
                             name='Quantile-based Clipping',
                             marker_color = 'orange',
                             showlegend=False,
                             marker_symbol=maks[2]),
                row=2, col=1)
    
    # Update subplot layout
    # fig.update_xaxes(title_text='Epsilon', row=1, col=1)
    fig.update_yaxes(title_text='RMSE', row=1, col=1)
    fig.update_xaxes(title_text='Epsilon', row=2, col=1)
    fig.update_yaxes(title_text='MAE', row=2, col=1)

    # Update the title of the entire figure
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="left", x=0.65),
                      height=800, 
                      width=800, 
                      title_text='<b>Comparison of Concentration Algorithms w/ Best-Fit Grouping</b>',
                      title_x=0.5)

    # fig.show() 
    return fig

def get_subplots_SampleScaling(
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
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.06, 
                        subplot_titles=("Array Averaging",
                                        ""
                                        )
                        )

#for reference only, DO NOT UNCOMMENT
# [
#  1       losses_base2_rmse,
#  2       losses_base2_bf_rmse,
#  3       losses_cm_rmse,
#  4       losses_cm_bf_rmse,
#  5       losses_q_wrap_rmse,
#  6       losses_q_best_rmse,
#  7       losses_base2_worst,
#  8       losses_base2_bf_worst,
#  9       losses_cm_worst,
#  10      losses_cm_bf_worst,
#  11      losses_q_wrap_worst,
#  12      losses_q_best_worst,
#     ]

    #creating an array of y values 
    y = []
    for i in y_axis_data:
        y.append(i)
    # print(y)
    # print(x_axis_data)

    # Add traces to the first subplot
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[3], 
                             mode='lines+markers', 
                             name='Wrap-around grouping',
                             marker_color = 'green',
                             marker_symbol=maks[0]),
                row=1, col=1)  
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[4], 
                             mode='lines+markers', 
                             name='Best-fit grouping',
                             marker_color = 'blue', 
                             marker_symbol=maks[1]),
                row=1, col=1)

    # Add a trace to the second subplot (Another Plot)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[9], 
                             mode='lines+markers', 
                             name='Wrap-around grouping',
                             marker_color = 'green',
                             showlegend=False,
                             marker_symbol=maks[0]),
                row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis_data, 
                             y=y[10], 
                             mode='lines+markers', 
                             name='Best-fit grouping',
                             marker_color = 'blue',
                             showlegend=False,
                             marker_symbol=maks[1]),
                row=2, col=1)
        
    # Update subplot layout
    # fig.update_xaxes(title_text='Epsilon', row=1, col=1)
    fig.update_yaxes(title_text='RMSE', row=1, col=1)
    fig.update_xaxes(title_text='Epsilon', row=2, col=1)
    fig.update_yaxes(title_text='MAE', row=2, col=1)

    # Update the title of the entire figure
    fig.update_layout(legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.7),
                      height=800, 
                      width=800, 
                      title_text='<b>Comparison of Grouping Algorithms for Non-Synthetic Data</b>',
                      title_x=0.5)

    # fig.show() 
    return fig


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

file_path_base = "./results_UserScaling/{}/{}/{}/losses.npy"
conc_algos = ["baseline", "coarse_mean", "quantiles"]
epsilons = config["epsilons"]
factor1 = 30
factor2 = 22

# losses_base_rmse = []
losses_base2_rmse = []
losses_base2_bf_rmse = []
losses_cm_rmse = []
losses_cm_bf_rmse = []
losses_q_wrap_rmse = []
losses_q_best_rmse = []

# losses_base_worst = []
losses_base2_worst = []
losses_base2_bf_worst = []
losses_cm_worst = []
losses_cm_bf_worst = []
losses_q_wrap_worst = []
losses_q_best_worst = []

for e in epsilons:
    # losses_base_e = np.load(file_path_base.format("baseline", "wrap", str(e)))
    losses_base2_e = np.load(file_path_base.format("baseline2", "wrap", str(e)))
    losses_base2_bf_e = np.load(file_path_base.format("baseline2", "best_fit", str(e)))
    losses_cm_e = np.load(file_path_base.format("coarse_mean", "wrap", str(e)))
    losses_cm_bf_e = np.load(file_path_base.format("coarse_mean", "best_fit", str(e)))
    losses_q_wrap_e = np.load(file_path_base.format("quantiles", "wrap", str(e)))
    losses_q_best_e = np.load(file_path_base.format("quantiles", "best_fit", str(e)))

    loaded_set = [
        # losses_base_e,
        losses_base2_e,
        losses_base2_bf_e,
        losses_cm_e,
        losses_cm_bf_e,
        losses_q_wrap_e,
        losses_q_best_e,
    ]
    store_set_rmse = [
        # losses_base_rmse,
        losses_base2_rmse,
        losses_base2_bf_rmse,
        losses_cm_rmse,
        losses_cm_bf_rmse,
        losses_q_wrap_rmse,
        losses_q_best_rmse,
    ]
    store_set_worst = [
        # losses_base_worst,
        losses_base2_worst,
        losses_base2_bf_worst,
        losses_cm_worst,
        losses_cm_bf_worst,
        losses_q_wrap_worst,
        losses_q_best_worst,
    ]

    for loaded_loss, store_array in zip(loaded_set, store_set_rmse):
        dim_rmses = np.sqrt(np.mean(loaded_loss**2, axis=1))
        # print(dim_rmses)
        overall_rmse = np.sqrt(np.mean(dim_rmses**2))
        # print(overall_rmse)
        store_array.append(overall_rmse)

    for loaded_loss, store_array in zip(loaded_set, store_set_worst):
        exp_perc = np.percentile(loaded_loss, 100, axis=0)
        # print(loaded_loss)
        # print(exp_perc)
        mean_perc = np.mean(exp_perc)
        store_array.append(mean_perc)


# fig = get_figure(
#     epsilons,
#     "Epsilon",
#     [
#         # losses_base_rmse,
#         losses_base2_rmse,
#         losses_base2_bf_rmse,
#         losses_cm_rmse,
#         losses_cm_bf_rmse,
#         losses_q_wrap_rmse,
#         losses_q_best_rmse,
#     ],
#     "Error",
#     "Error vs Epsilon",
#     [
#         # "Baseline",
#         "Array Averaging + wrap-around grouping",
#         "Array Averaging + best-fit grouping",
#         "Levy Algorithm + wrap-around grouping",*
#         "Levy Algorithm + best-fit grouping",*
#         "DAPaMean-MD + wrap-around grouping",
#         "DAPaMean-MD + best-fit grouping",
#     ],
#     legend_prefix="",
# )

# fig.show()

# fig = get_figure(
#     epsilons,
#     "Epsilon",
#     [
#         # losses_base_worst,
#         losses_base2_worst,
#         losses_base2_bf_worst,
#         losses_cm_worst,
#         losses_cm_bf_worst,
#         losses_q_wrap_worst,
#         losses_q_best_worst,
#     ],
#     "Error",
#     "Error vs Epsilon",
#     [
#         # "Baseline",
#         "AAA + wrap",
#         "AAA + best",
#         "Levy + wrap",
#         "Levy + best",
#         "DAPaMean-MD wrap",
#         "DAPaMean-MD best",
#     ],
#     legend_prefix="",
# )

# fig.show()

fig = get_subplots_nonSynthetic(
    epsilons,
    "Epsilon",
    [
        losses_base2_rmse,
        losses_base2_bf_rmse,
        losses_cm_rmse,
        losses_cm_bf_rmse,
        losses_q_wrap_rmse,
        losses_q_best_rmse,
        losses_base2_worst,
        losses_base2_bf_worst,
        losses_cm_worst,
        losses_cm_bf_worst,
        losses_q_wrap_worst,
        losses_q_best_worst,
    ],
    "Error",
    "Error vs Epsilon",
    [
        # "Baseline",
        "AAA + wrap",
        "AAA + best",
        "Levy + wrap",
        "Levy + best",
        "DAPaMean-MD wrap",
        "DAPaMean-MD best",
    ],
    legend_prefix="",
)

fig.write_image("results_UserScaling/plots/ConcentrationAlgoComparisonBestFit_US.png")














# fig.show()
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
