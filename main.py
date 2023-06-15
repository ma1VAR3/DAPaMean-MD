import json
import os

from utils import calc_dim_qtile, calc_dim_qtile_dropping, calc_support

from groupping import get_user_arrays
from estimation import private_estimation, baseline_estimation

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile)
        print("Configurations loaded from config.json")
        jsonfile.close()

    dataset = config["dataset"]
    data = pd.read_csv("./data/synthetic_data.csv")
    metadata = pd.read_csv("./data/synthetic_metadata.csv")

    dims = data["Dimension"].unique()
    users = data[
        "User"
    ].unique()  # <- List of users, used everywhere so that order is maintained

    dim_qtile_base = calc_dim_qtile(data, 100)

    if config["drop_dims"] == True:
        support = calc_support(metadata)
        print("Support: ", support)
        dim_qtile_pass1 = calc_dim_qtile_dropping(data, metadata, users, support, 90, 1)
        dim_qtile, data, metadata = calc_dim_qtile_dropping(
            data, metadata, users, support, dim_qtile_pass1, 2
        )
        print("dim_qtile from pass 1: ", dim_qtile_pass1)
        print("dim_qtile from pass 2: ", dim_qtile)

    epsilons = config["epsilons"]

    print("Starting experiments")

    if config["concentration_algorithm"] == "baseline":
        for e in epsilons:
            print("For epsilon: ", e)
            exp_err = []
            for prog, d in zip(tqdm(range(len(dims))), dims):
                # for d in dims:
                d_data = data[data["Dimension"] == d]
                upper_bound = config["data"][dataset]["upper_bound"]
                lower_bound = config["data"][dataset]["lower_bound"]
                num_experiments = config["num_experiments"]
                actual_mean = metadata[metadata["Dimension"] == d][
                    "Actual Mean"
                ].values[0]
                dim_losses = baseline_estimation(
                    d_data,
                    upper_bound,
                    lower_bound,
                    e,
                    actual_mean,
                    num_experiments,
                )
                exp_err.append(np.array(dim_losses))
                # print(sys.getsizeof(np.array(exp_err)))
            os.makedirs(
                "./results/"
                + config["concentration_algorithm"]
                + "/"
                + config["user_groupping"]
                + "/"
                + str(e)
                + "/",
                exist_ok=True,
            )

            np.save(
                "./results/"
                + config["concentration_algorithm"]
                + "/"
                + config["user_groupping"]
                + "/"
                + str(e)
                + "/"
                + "losses.npy",
                exp_err,
            )

    else:
        for e in epsilons:
            print("For epsilon: ", e)
            exp_err = []
            for prog, d in zip(tqdm(range(len(dims))), dims):
                d_data = data[data["Dimension"] == d]
                L = metadata[metadata["Dimension"] == d]["L"].values[0]
                user_arrays, K = get_user_arrays(d_data, L, config["user_groupping"])
                actual_mean = metadata[metadata["Dimension"] == d][
                    "Actual Mean"
                ].values[0]
                user_group_means = [np.mean(x) for x in user_arrays]
                dim_losses = private_estimation(
                    user_group_means,
                    L,
                    K,
                    config["data"][dataset]["upper_bound"],
                    config["data"][dataset]["lower_bound"],
                    e,
                    config["num_experiments"],
                    actual_mean,
                    config["user_groupping"],
                    config["concentration_algorithm"],
                    config["algorithm_parameters"][config["concentration_algorithm"]],
                )
                exp_err.append(np.array(dim_losses))

            os.makedirs(
                "./results/"
                + config["concentration_algorithm"]
                + "/"
                + config["user_groupping"]
                + "/"
                + str(e)
                + "/",
                exist_ok=True,
            )

            np.save(
                "./results/"
                + config["concentration_algorithm"]
                + "/"
                + config["user_groupping"]
                + "/"
                + str(e)
                + "/"
                + "losses.npy",
                exp_err,
            )
