import json
import os

from utils import calc_dim_qtile, calc_dim_qtile_dropping, calc_support

from groupping import get_user_arrays
from estimation import private_estimation, baseline_estimation

import numpy as np
import pandas as pd

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile) 
        print("Configurations loaded from config.json")
        jsonfile.close()
    dataset = config["dataset"]
    data = pd.read_csv("./data/filtered_data.csv")
    metadata = pd.read_csv("./data/filtered_metadata.csv")
    
    """
    Step 1: Calculate dim_qtile qth ptile of num_dims contrib by a user
    Step 2: Calculate k* =  min k to protect      -> Can be implicitly guarenteed? -> Replace metric? root(lk)
    Step 3: Iterate through users contrib > dim*
        SubStep 1: For the given user, drop samples from the dimension with max k, and check if new k > k*
        SubStep 2: If yes, keep dropping samples until contrib = dim* or new k < k*
        SubStep 3: If new k < k* stop dropping samples
    Step 4: eps for each dim = epsilon / dim*
    Step 5: Run DAPaMean-MD for each dim
    
    
    NOTE: This is a greedy algorithm, and may not be optimal. Optimize to maximize root(lk)
        -> L can be mean/some percentile of number of samples
    """    
    
    
    dims = data["Dimension"].unique()
    users = data["User"].unique() # <- List of users, used everywhere so that order is maintained
    
    dim_qtile_base = calc_dim_qtile(data, 100)
    
    if config["concentration_algorithm"] == "quantiles":
        support = calc_support(metadata)
        print("Support: ", support)
        dim_qtile_pass1 = calc_dim_qtile_dropping(data, metadata, users, support, 90, 1)
        dim_qtile, data, metadata = calc_dim_qtile_dropping(data, metadata, users, support, dim_qtile_pass1, 2)
        print("dim_qtile from pass 1: ", dim_qtile_pass1)
        print("dim_qtile from pass 2: ", dim_qtile)
        # for u in users:
        #     user_data = data[data["User"]==u]
        #     num_user_dims = len(user_data["Dimension"].unique())
            
        #     if num_user_dims > dim_qtile:
        #         user_data = data[data["User"]==u]
        #         k_ordered_dims = get_k_ordered_dims(user_data, metadata)
        #         for s_dim in k_ordered_dims:
        #             if num_user_dims > dim_qtile:
        #                 new_data, new_metadata = drop_dim_for_user(data, metadata, s_dim, u)
        #                 support_post_drop = calc_support(new_metadata)
        #                 if support_post_drop >= support:
        #                     data = new_data
        #                     metadata = new_metadata
        #                     num_user_dims -= 1
        #             else:
        #                 break
                    
        #     user_data = data[data["User"]==u]
        #     num_user_dims = len(user_data["Dimension"].unique())
            
        #     if num_user_dims > dim_qtile:
        #         dim_qtile = num_user_dims
        # data.to_csv('./dropped_data.csv', index=False)
        
            
        
    
    algo_err = []
    epsilons = config["epsilons"]
    
    if config["concentration_algorithm"] == "baseline":
        for e in epsilons:
            # e_prime = e / dim_qtile_base
            exp_err = 0
            for d in dims:
                d_data = data[data["Dimension"]==d]
                upper_bound = config["data"][dataset]["upper_bound"]
                lower_bound = config["data"][dataset]["lower_bound"]
                num_experiments = config["num_experiments"]
                actual_mean = metadata[metadata["Dimension"]==d]["Actual Mean"].values[0]
                dim_rmse = baseline_estimation(d_data, upper_bound, lower_bound, e, actual_mean, num_experiments)
                exp_err += dim_rmse * dim_rmse
            exp_err = np.sqrt(exp_err / len(dims))
            algo_err.append(exp_err)
    else:
        for e in epsilons:
            # if config["concentration_algorithm"] == "quantiles":
            #     e_prime = e / dim_qtile
            # else:
            #     e_prime = e / dim_qtile_base
            exp_err = 0
            for d in dims:
                print(d)
                d_data = data[data["Dimension"]==d]
                L = metadata[metadata["Dimension"]==d]["L"].values[0]
                user_arrays, K = get_user_arrays(d_data, L, config["user_groupping"])
                actual_mean = metadata[metadata["Dimension"]==d]["Actual Mean"].values[0]
                user_group_means = [np.mean(x) for x in user_arrays]
                dim_rmse = private_estimation(
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
                    config["algorithm_parameters"][config["concentration_algorithm"]]
                )
                exp_err += dim_rmse * dim_rmse
            exp_err = np.sqrt(exp_err / len(dims))
            algo_err.append(exp_err)
    print("Exp errs: ", algo_err)
    os.makedirs('./new_res/' + config["concentration_algorithm"] + '/', exist_ok=True)
    np.save('./new_res/' + config["concentration_algorithm"] + '/' + 'losses.npy', algo_err)
    