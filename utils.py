import json
import os
import time
import math
import numpy as np
import pandas as pd
import h3
from tqdm import tqdm
import matplotlib.pyplot as plt


def std_preproc_itms(config):
    df = pd.read_csv("./data/suratITMSDPtest.csv")
    df = df.drop_duplicates(
        subset=["trip_id", "observationDateTime"], ignore_index=True
    )
    df = df.drop(
        columns=[
            "trip_direction",
            "last_stop_id",
            "last_stop_arrival_time",
            "route_id",
            "actual_trip_start_time",
            "trip_delay",
            "vehicle_label",
            "id",
            "location.type",
            "trip_id",
        ]
    )
    lat_lon = df["location.coordinates"].astype(str).str.strip("[]").str.split(",")
    lon = lat_lon.apply(lambda x: x[0])
    lat = lat_lon.apply(lambda x: x[1])
    dflen = len(df)
    h3index = [None] * dflen
    resolution = config["H3 Resolution"]
    for i in range(dflen):
        h3index[i] = h3.geo_to_h3(
            lat=float(lat[i]), lng=float(lon[i]), resolution=resolution
        )
    df["h3index"] = h3index
    df["Date"] = pd.to_datetime(df["observationDateTime"]).dt.date
    df["Time"] = pd.to_datetime(df["observationDateTime"]).dt.time
    time = df["Time"]
    df["Timeslot"] = time.apply(lambda x: x.hour)
    df["HAT"] = df["Timeslot"].astype(str) + " " + df["h3index"]
    startTime = config["Start time"]
    endTime = config["End time"]
    df = df[(df["Timeslot"] >= startTime) & (df["Timeslot"] <= endTime)]
    df = df[df["speed"] > 0]

    df = df.drop(
        columns=[
            "observationDateTime",
            "location.coordinates",
            "h3index",
            "Date",
            "Time",
            "Timeslot",
        ]
    )
    df = df.rename(
        columns={"license_plate": "User", "speed": "Value", "HAT": "Dimension"}
    )

    num_user_filtering_df = (
        df.groupby(["Dimension"]).agg({"User": "nunique"}).reset_index()
    )
    num_user_filtering_df = num_user_filtering_df.sort_values(
        by=["User"], ascending=False
    ).reset_index()
    num_user_filtering_df = num_user_filtering_df.loc[0]
    chosen_hat = num_user_filtering_df["Dimension"]

    print(chosen_hat)
    df = df[df["Dimension"] == chosen_hat]

    return df


def generate_metadata(data):
    dims = data["Dimension"].unique()
    l_vals = []
    k_vals = []
    actual_mean_vals = []
    for d in dims:
        dim_data = data[data["Dimension"] == d]
        dim_l = calc_user_array_length(dim_data)
        l_vals.append(dim_l)
        dim_k = calc_k(dim_data, dim_l)
        k_vals.append(dim_k)
        dim_vals = dim_data["Value"].values
        dim_actual_mean = np.mean(dim_vals)
        actual_mean_vals.append(dim_actual_mean)
    metadata_dict = {
        "Dimension": dims,
        "L": l_vals,
        "K": k_vals,
        "Actual Mean": actual_mean_vals,
    }
    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df["Sup"] = np.sqrt(metadata_df["L"]) * metadata_df["K"]
    return metadata_df


def load_data(preproc_data, num_exp=0, config=None):
    data = None
    if config["Synthetic"] == False:
        return preproc_data, generate_metadata(preproc_data)

    if config["Synthetic"] == True:
        df = preproc_data
        data = None

        os.makedirs(
            "./gen_data/",
            exist_ok=True,
        )
        for i in range(num_exp):
            if config["Synthetic Scaling"] == "samples":
                data = synthesize_perdim_scale_samples(df, config)
            elif config["Synthetic Scaling"] == "users":
                data = synthesize_perdim_scale_users(df, config)
            else:
                print("Invalid scaling type")
            data.to_csv("./gen_data/" + str(i) + ".csv", index=False)
            metadata = generate_metadata(data)
            metadata.to_csv("./gen_data/" + str(i) + "_metadata.csv", index=False)



def synthesize_perdim_peruser(data, config):
    dimension_arr = []
    user_arr = []
    value_arr = []
    bins = np.arange(config["lower_bound"], config["upper_bound"] + 1, 1)
    dims = data["Dimension"].unique()
    for d in dims:
        # for d in dims:
        dim_data = data[data["Dimension"] == d]
        dim_users = dim_data["User"].unique()
        for u in dim_users:
            dim_user_data = dim_data[dim_data["User"] == u]
            num_user_samples = len(dim_user_data) * config["Synthetic Factor"]
            user_dim_samples = dim_user_data["Value"].values
            user_new_samples = []
            hist = np.histogram(user_dim_samples, bins=bins)
            bin_probs = hist[0] / np.sum(hist[0])
            for x in range(num_user_samples):
                rand_bin = np.random.choice(bins[:-1], p=bin_probs)
                rand_sample = np.random.uniform(rand_bin, rand_bin + 1)
                dimension_arr.append(d)
                user_arr.append(u)
                value_arr.append(rand_sample)
                user_new_samples.append(rand_sample)
            # plt.hist(user_dim_samples, bins=bins, density=True)
            # plt.show()
            # plt.hist(user_new_samples, bins=bins, density=True)
            # plt.show()
        value_arr = np.clip(value_arr, config["lower_bound"], config["upper_bound"])
    syn_data = pd.DataFrame(
        {"Dimension": dimension_arr, "User": user_arr, "Value": value_arr}
    )

    return syn_data


def synthesize_perdim_scale_samples(data, config):
    dimension_arr = []
    user_arr = []
    value_arr = []
    dims = data["Dimension"].unique()
    for d in dims:
        dim_data = data[data["Dimension"] == d]
        dim_samples = dim_data["Value"].values
        mean = np.mean(dim_samples)
        std = np.std(dim_samples)
        dim_users = dim_data["User"].unique()
        num_dim_samples = len(dim_samples) * config["Synthetic Factor"]
        start = time.time()
        new_dim_samples = np.random.normal(mean, std, num_dim_samples)
        end = time.time()
        print("Sampling time: ", end - start)
        value_arr.extend(new_dim_samples)
        for u in dim_users:
            num_user_samples_in_dim = len(
                dim_data[dim_data["User"] == u]["Value"].values
            )
            dimension_arr.extend(
                [d] * (num_user_samples_in_dim * config["Synthetic Factor"])
            )
            user_arr.extend(
                [u] * (num_user_samples_in_dim * config["Synthetic Factor"])
            )
        value_arr = np.clip(value_arr, config["lower_bound"], config["upper_bound"])
    syn_data = pd.DataFrame(
        {"Dimension": dimension_arr, "User": user_arr, "Value": value_arr}
    )
    return syn_data


def synthesize_perdim_scale_users(data, config):
    dimension_arr = []
    user_arr = []
    value_arr = []
    dims = data["Dimension"].unique()
    for d in dims:
        dim_data = data[data["Dimension"] == d]
        dim_samples = dim_data["Value"].values
        mean = np.mean(dim_samples)
        std = np.std(dim_samples)
        dim_users = dim_data["User"].unique()
        for u in dim_users:
            num_user_samples_in_dim = len(
                dim_data[dim_data["User"] == u]["Value"].values
            )
            for x in range(1, config["Synthetic Factor"] + 1):
                new_uid = u + "_" + str(x)
                new_user_samples = np.random.normal(mean, std, num_user_samples_in_dim)
                dimension_arr.extend([d] * num_user_samples_in_dim)
                user_arr.extend([new_uid] * num_user_samples_in_dim)
                value_arr.extend(new_user_samples)
        value_arr = np.clip(value_arr, config["lower_bound"], config["upper_bound"])
        # plt.hist(value_arr, density=True, bins=65)
        # plt.show()
        # plt.hist(dim_samples, bins=bins, density=True)
        # plt.show()
        # plt.hist(new_samples, bins=bins, density=True)
        # plt.show()

    syn_data = pd.DataFrame(
        {"Dimension": dimension_arr, "User": user_arr, "Value": value_arr}
    )
    return syn_data


def calc_k(data, l):
    """
    Takes data for one dimension and calculates corresponding k
    """
    data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    k = np.sum([np.minimum(i, l) for i in data_grouped["Value"]]) / l
    return k


def calc_dim_qtile(data, qtile):
    # unique_dims = data["Dimension"].unique()
    # print("ALL DIMS: ", len(unique_dims))
    data_gb_user = data.groupby(["User"]).agg({"Dimension": "nunique"}).reset_index()
    dims = data_gb_user["Dimension"].values
    # users = data["User"].unique()
    # dims = []
    # for u in users:
    #     user_data = data[data["User"]==u]
    #     dims.append(len(user_data["Dimension"].unique()))

    # import plotly.express as px
    # fig = px.histogram(dims)
    # fig.show()

    dim_qtile = np.percentile(dims, qtile)
    # print("Max dim contributed: ", np.max(dims))
    print("{}th percentile of dims contributed: ".format(qtile), dim_qtile)
    return dim_qtile


def get_sup_ordered_dims(user_data, metadata):
    """
    Get the k ordered dimensions for a user
    """
    user_dims = user_data["Dimension"].unique()
    user_metadata = metadata[metadata["Dimension"].isin(user_dims)]
    user_metadata = user_metadata.sort_values(by=["Sup"], ascending=False)
    k_ordered_dims = user_metadata["Dimension"].values
    return k_ordered_dims


def calc_dim_qtile_dropping(data, metadata, users, support, qtile, pass_num):
    """
    The two pass approach to calculate threhold for dropping dimensions
    """
    if pass_num == 1:
        dim_qtile = calc_dim_qtile(data, qtile)  # <- Initial threshold
    else:
        dim_qtile = qtile

    print("Starting pass ", pass_num)

    for prog, u in zip(tqdm(range(len(users))), users):
        user_data = data[data["User"] == u]
        num_user_dims = len(user_data["Dimension"].unique())

        if num_user_dims > dim_qtile:
            user_data = data[
                data["User"] == u
            ]  # <- Get user data again (since dims have been dropped)
            sup_ordered_dims = get_sup_ordered_dims(user_data, metadata)
            for s_dim in sup_ordered_dims:
                if num_user_dims > dim_qtile:
                    new_data, new_metadata = drop_dim_for_user(data, metadata, s_dim, u)
                    support_post_drop = calc_support(new_metadata)
                    if support_post_drop >= support:
                        data = new_data
                        metadata = new_metadata
                        num_user_dims -= 1
                else:
                    break

        user_data = data[data["User"] == u]
        num_user_dims = len(user_data["Dimension"].unique())

        if num_user_dims > dim_qtile:
            dim_qtile = num_user_dims

    dim_qtile = calc_dim_qtile(data, 100)  # <- Final threshold
    if pass_num == 1:
        return dim_qtile
    else:
        data.to_csv("./dropped_data.csv", index=False)
        return dim_qtile, data, metadata


def calc_support(metadata):
    sup_vals = metadata["Sup"].values

    # dims = data["Dimension"].unique()
    # k_vals = []
    # for d in dims:
    #     dim_data = data[data["Dimension"]==d]
    #     dim_l = calc_user_array_length(dim_data)
    #     dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    #     dim_k = np.floor(np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l)
    #     k_vals.append(dim_k)

    # import plotly.express as px
    # fig = px.histogram(k_vals)
    # fig.show()

    # print("Min k: ", np.min(k_vals))

    return np.min(sup_vals)


# def get_max_k_dim(user_data, data):
#     """
#     Gets the dimension with the maximum k value for the given user data
#     """
#     user_dims = user_data["Dimension"].unique()
#     max_user_k_val = -1
#     max_k_val_dim = None
#     for d in user_dims:
#         dim_data = data[data["Dimension"]==d]
#         dim_l = calc_user_array_length(dim_data)
#         dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
#         dim_k = np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l
#         if dim_k > max_user_k_val:
#             max_user_k_val = dim_k
#             max_k_val_dim = d
#     return max_k_val_dim

# def get_support_post_drop(data, dim, user):
#     """
#     Gets the k value for the given dimension after dropping the dimension for the given user
#     """
#     new_data = drop_dim_for_user(data, dim, user)
#     dim_data = new_data[new_data["Dimension"]==dim]
#     dim_l = calc_user_array_length(dim_data)
#     dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
#     dim_k = np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l
#     return dim_k


def drop_dim_for_user(data, metadata, dim, user):
    """
    Drops the rows corresponding to the given dimension for the given user
    """

    entries_to_drop = data[(data["Dimension"] == dim) & (data["User"] == user)]
    new_data = data.drop(entries_to_drop.index, inplace=False)
    new_k_for_dim = calc_k(
        new_data[new_data["Dimension"] == dim],
        metadata[metadata["Dimension"] == dim]["L"].values[0],
    )
    metadata.loc[metadata["Dimension"] == dim, "K"] = new_k_for_dim
    metadata.loc[metadata["Dimension"] == dim, "Sup"] = (
        np.sqrt(metadata[metadata["Dimension"] == dim]["L"].values[0]) * new_k_for_dim
    )
    return new_data, metadata


def target_funtion(l, user_contribs):
    """
    The target function for the optimisation problem
    """
    k = np.sum([np.minimum(i, l) for i in user_contribs]) / l
    return np.sqrt(l) * k


def optimize_me(user_contribs, start, end, step=1):
    f_val = float("-inf")
    l_opt = start
    for l in range(start, end, step):
        val = target_funtion(l, user_contribs)
        if val > f_val:
            f_val = val
            l_opt = l
    return l_opt


def calc_user_array_length(data, type="opt"):
    L = None
    data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    user_contribs = data_grouped["Value"].values

    if type == "median":
        L = math.floor(np.median(data_grouped["Value"]))
    elif type == "mean":
        L = math.floor(np.mean(data_grouped["Value"]))
    elif type == "max":
        L = np.max(data_grouped["Value"])
    elif type == "rms":
        L = math.floor(
            math.sqrt(np.mean([math.pow(i, 2) for i in data_grouped["Value"]]))
        )
    elif type == "opt":
        start = np.percentile(user_contribs, 10).astype(int)
        end = np.percentile(user_contribs, 90).astype(int)
        L = optimize_me(user_contribs, start, end, step=1)
    return L


def comp_arrays(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    len1a = len(arr1[0])
    len2a = len(arr2[0])
    print("Dimensions of array 1: ", len1, ", ", len1a)
    print("Dimensions of array 2: ", len2, ", ", len2a)
    flag = True
    fault_counts = 0
    if len1 == len2 and len1a == len2a:
        for x, y in zip(arr1, arr2):
            for x1 in x:
                if x1 not in y:
                    fault_counts += 1
                    flag = False
    print("faults:", fault_counts)
    return flag


def comp_arrays_1d(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    flag = True
    faults = 0
    fault_idx = []
    if len1 != len2:
        flag = False
        return flag
    for i in range(len1):
        if round(arr1[i], 4) != round(arr2[i], 4):
            flag = False
            faults += 1
            fault_idx.append(i)
    print("faults:", faults)
    # print(arr1)
    # print(arr2)
    # print(fault_idx)
    return flag


def comp_dicts(dict1, dict2):
    flag = True
    for dict1_key in dict1.keys():
        if dict1_key not in dict2.keys():
            print(dict1_key)
            flag = False
            break
        dict1_user_dict = dict1[dict1_key]
        dict2_user_dict = dict2[dict1_key]
        if dict1_user_dict != dict2_user_dict:
            flag = False
            break
    if flag == True:
        print("Metadata match!")
    else:
        print("Metadata DO NOT match!!!!")
