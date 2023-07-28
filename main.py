import json
import os
import time

from utils import (
    calc_dim_qtile,
    calc_dim_qtile_dropping,
    calc_support,
    load_data,
    comp_arrays,
    comp_dicts,
    comp_arrays_1d,
)

from groupping import (
    get_user_arrays,
    get_weighted_mean_template,
    get_array_mean_from_template,
)
from estimation import private_estimation, baseline_estimation

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile)
        print("Configurations loaded from config.json")
        jsonfile.close()

    dataset = config["dataset"]
    preproc_data = pd.read_csv("./data/ITMS_processed.csv")
    dims = preproc_data["Dimension"].unique()

    epsilons = config["epsilons"]
    print("Starting experiments")

    # conc_algorithms = ["baseline2", "coarse_mean", "quantiles"]
    # groupping_algos = ["wrap", "best_fit"]
    conc_algo = config["concentration_algorithm"]
    group_algo = config["user_groupping"]

    # for conc_algo in conc_algorithms:
    print("=" * 20 + " CONC ALGO: {}".format(conc_algo) + "=" * 20)
    # for group_algo in groupping_algos:
    print("-" * 20 + " Groupping Algo: {}".format(group_algo) + "-" * 20)

    for e in epsilons:
        print("-" * 20 + " Epsilon: {}".format(e) + "-" * 20)
        exp_err = []
        for prog, d in zip(tqdm(range(len(dims))), dims):
            # print("Starting exps")
            num_exps = config["num_experiments"]
            dim_losses = []
            data, metadata = None, None
            array_template, K = None, None

            if config["data"][dataset]["Synthetic"] == False:
                data, metadata = load_data(preproc_data, config=config["data"][dataset])
                # print("Init data loaded")
                d_data = data[data["Dimension"] == d]
                L = metadata[metadata["Dimension"] == d]["L"].values[0]
                array_template, K = get_weighted_mean_template(
                    d_data, L, config["user_groupping"]
                )

            else:
                data = pd.read_csv("./gen_data/" + str(0) + ".csv")
                metadata = pd.read_csv("./gen_data/" + str(0) + "_metadata.csv")
                d_data = data[data["Dimension"] == d]
                L = metadata[metadata["Dimension"] == d]["L"].values[0]
                array_template, K = get_weighted_mean_template(
                    d_data, L, config["user_groupping"]
                )

            for exp in range(num_exps):
                if config["data"][dataset]["Synthetic"] == True:
                    # data, metadata = load_data(
                    #     preproc_data, config["data"][dataset]
                    # )
                    data = pd.read_csv("./gen_data/" + str(exp) + ".csv")
                    metadata = pd.read_csv("./gen_data/" + str(exp) + "_metadata.csv")

                d_data = data[data["Dimension"] == d]
                L = metadata[metadata["Dimension"] == d]["L"].values[0]

                # user_array_template, K = create_array_template(
                #     d_data, L, config["user_groupping"]
                # )
                # array_template_weights, K_weights = get_weighted_mean(
                #     d_data, L, config["user_groupping"]
                # )
                # comp_dicts(user_array_template, array_template_weights)
                # user_arrays_new = get_user_arrays_new(
                #     d_data, L, K, user_array_template, config["user_groupping"]
                # )
                # user_arrays, K = get_user_arrays(d_data, L, config["user_groupping"])
                # print(len(user_arrays_new))

                # res = comp_arrays(user_arrays, user_arrays_new)
                # if res == True:
                #     print("Grouping successful!")
                # else:
                #     print("Grouping unsuccessful")
                # np.save("./array_template.npy", user_array_template)

                actual_mean = metadata[metadata["Dimension"] == d][
                    "Actual Mean"
                ].values[0]

                # group_time_st = time.time()
                user_group_means = get_array_mean_from_template(
                    array_template, K, d_data
                )
                # group_time_end = time.time()
                # print("Groupping time: ", group_time_end - group_time_st)
                # user_group_means = [np.mean(x) for x in user_arrays]
                # res = comp_arrays_1d(user_group_means, user_group_means_new)
                # plt.hist(user_group_means, density=True, bins=110)
                # plt.show()
                # start = time.time()
                dim_loss = private_estimation(
                    user_group_means,
                    L,
                    K,
                    config["data"][dataset]["upper_bound"],
                    config["data"][dataset]["lower_bound"],
                    e,
                    1,
                    actual_mean,
                    group_algo,
                    conc_algo,
                    config["algorithm_parameters"][conc_algo],
                )
                # end = time.time()
                # print("Estimation time:", end - start)
                dim_losses.extend(dim_loss.tolist())
                # tot_end = time.time()
                # print("One shot time: ", tot_end - tot_start)
            exp_err.append(np.array(dim_losses))

        os.makedirs(
            "./results/" + conc_algo + "/" + group_algo + "/" + str(e) + "/",
            exist_ok=True,
        )

        np.save(
            "./results/"
            + conc_algo
            + "/"
            + group_algo
            + "/"
            + str(e)
            + "/"
            + "losses.npy",
            exp_err,
        )

    # users = data[
    #     "User"
    # ].unique()  # <- List of users, used everywhere so that order is maintained

    # dim_qtile_base = calc_dim_qtile(data, 100)

    # if config["drop_dims"] == True and not os.path.exists("./data/dropped_data.csv"):
    #     support = calc_support(metadata)
    #     print("Support: ", support)
    #     dim_qtile_pass1 = calc_dim_qtile_dropping(data, metadata, users, support, 90, 1)
    #     dim_qtile, data, metadata = calc_dim_qtile_dropping(
    #         data, metadata, users, support, dim_qtile_pass1, 2
    #     )
    #     print("dim_qtile from pass 1: ", dim_qtile_pass1)
    #     print("dim_qtile from pass 2: ", dim_qtile)
    #     data.to_csv("./data/dropped_data.csv")
    #     metadata.to_csv("./data/dropped_metadata.csv")

    # if config["concentration_algorithm"] == "baseline":
    #     for e in epsilons:
    #         print("For epsilon: ", e)
    #         exp_err = []
    #         for prog, d in zip(tqdm(range(len(dims))), dims):
    #             dim_losses = []
    #             num_experiments = config["num_experiments"]
    #             dataset = config["dataset"]
    #             if config["data"][dataset]["Synthetic"] == False:
    #                 data, metadata = load_data(preproc_data, config["data"][dataset])
    #             for exp in range(num_experiments):
    #                 if config["data"][dataset]["Synthetic"] == True:
    #                     data, metadata = load_data(
    #                         preproc_data, config["data"][dataset]
    #                     )
    #                 # for d in dims:
    #                 d_data = data[data["Dimension"] == d]
    #                 upper_bound = config["data"][dataset]["upper_bound"]
    #                 lower_bound = config["data"][dataset]["lower_bound"]
    #                 actual_mean = metadata[metadata["Dimension"] == d][
    #                     "Actual Mean"
    #                 ].values[0]
    #                 dim_loss = baseline_estimation(
    #                     d_data,
    #                     upper_bound,
    #                     lower_bound,
    #                     e,
    #                     actual_mean,
    #                     1,
    #                 )
    #                 dim_losses.extend(dim_loss.tolist())
    #             exp_err.append(np.array(dim_losses))
    #             # print(sys.getsizeof(np.array(exp_err)))
    #         os.makedirs(
    #             "./results/"
    #             + config["concentration_algorithm"]
    #             + "/"
    #             + config["user_groupping"]
    #             + "/"
    #             + str(e)
    #             + "/",
    #             exist_ok=True,
    #         )

    #         np.save(
    #             "./results/"
    #             + config["concentration_algorithm"]
    #             + "/"
    #             + config["user_groupping"]
    #             + "/"
    #             + str(e)
    #             + "/"
    #             + "losses.npy",
    #             exp_err,
    #         )

    # else:
    #     for e in epsilons:
    #         print("For epsilon: ", e)
    #         exp_err = []
    #         for prog, d in zip(tqdm(range(len(dims))), dims):
    #             num_exps = config["num_experiments"]
    #             dim_losses = []
    #             dataset = config["dataset"]
    #             if config["data"][dataset]["Synthetic"] == False:
    #                 data, metadata = load_data(preproc_data, config["data"][dataset])
    #             for exp in range(num_exps):
    #                 print(exp)
    #                 if config["data"][dataset]["Synthetic"] == True:
    #                     data, metadata = load_data(
    #                         preproc_data, config["data"][dataset]
    #                     )
    #                 d_data = data[data["Dimension"] == d]
    #                 L = metadata[metadata["Dimension"] == d]["L"].values[0]
    #                 user_arrays, K = get_user_arrays(
    #                     d_data, L, config["user_groupping"]
    #                 )
    #                 actual_mean = metadata[metadata["Dimension"] == d][
    #                     "Actual Mean"
    #                 ].values[0]
    #                 user_group_means = [np.mean(x) for x in user_arrays]
    #                 # plt.hist(user_group_means, density=True, bins=110)
    #                 # plt.show()
    #                 dim_loss = private_estimation(
    #                     user_group_means,
    #                     L,
    #                     K,
    #                     config["data"][dataset]["upper_bound"],
    #                     config["data"][dataset]["lower_bound"],
    #                     e,
    #                     1,
    #                     actual_mean,
    #                     config["user_groupping"],
    #                     config["concentration_algorithm"],
    #                     config["algorithm_parameters"][
    #                         config["concentration_algorithm"]
    #                     ],
    #                 )
    #                 dim_losses.extend(dim_loss.tolist())

    #             exp_err.append(np.array(dim_losses))

    #         os.makedirs(
    #             "./results/"
    #             + config["concentration_algorithm"]
    #             + "/"
    #             + config["user_groupping"]
    #             + "/"
    #             + str(e)
    #             + "/",
    #             exist_ok=True,
    #         )

    #         np.save(
    #             "./results/"
    #             + config["concentration_algorithm"]
    #             + "/"
    #             + config["user_groupping"]
    #             + "/"
    #             + str(e)
    #             + "/"
    #             + "losses.npy",
    #             exp_err,
    #         )
