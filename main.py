import json
import os

from utils import calc_dim_qtile, calc_dim_qtile_dropping, calc_support
from utils import get_k_ordered_dims, drop_dim_for_user
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
        dim_qtile = calc_dim_qtile_dropping(data, metadata, users, support, 90)
        for u in users:
            
            user_data = data[data["User"]==u]
            num_user_dims = len(user_data["Dimension"].unique())
            while num_user_dims > dim_qtile:
                user_data = data[data["User"]==u]
                # max_k_dim = get_max_k_dim(user_data, data)
                k_ordered_dims = get_k_ordered_dims(user_data, metadata)
                for s_dim in k_ordered_dims:
                    if metadata[metadata["Dimension"]==s_dim]["Sup"].values[0] < support:
                        break
                    elif metadata[metadata["Dimension"]==s_dim]["Sup"].values[0]-1 < support:
                        new_data, new_metadata = drop_dim_for_user(data, metadata, s_dim, u)
                        support_post_drop = calc_support(new_metadata)
                        if support_post_drop >= support:
                            num_user_dims -=1
                            data = new_data
                            metadata = new_metadata
                            break
                            # data, metadata = drop_dim_for_user(data, metadata, s_dim, u)
                    else:
                        num_user_dims -=1
                        data, metadata = drop_dim_for_user(data, metadata, s_dim, u)
                        break
                # if metadata[max_k_dim]["K"] -1 < support:
                #     support_post_drop = get_support_post_drop(data, max_k_dim, u)
                #     if support_post_drop < support:
                #         # dim_qtile = num_user_dims
                #         break
                #     else:
                #         num_user_dims -=1
                #         data, metadata = drop_dim_for_user(data, metadata, max_k_dim, u)
                # else:
                #     num_user_dims -=1
                #     data, metadata = drop_dim_for_user(data, metadata, max_k_dim, u)
        data.to_csv('./dropped_data.csv', index=False)
        
            
        dim_qtile_new = calc_dim_qtile(data, 100)
        print("dim_qtile from pass 1: ", dim_qtile)
        print("dim_qtile from pass 2: ", dim_qtile_new)
    
    algo_err = []
    epsilons = config["epsilons"]
    
    if config["concentration_algorithm"] == "baseline":
        for e in epsilons:
            e_prime = e / dim_qtile_base
            exp_err = 0
            for d in dims:
                d_data = data[data["Dimension"]==d]
                upper_bound = config["data"][dataset]["upper_bound"]
                lower_bound = config["data"][dataset]["lower_bound"]
                num_experiments = config["num_experiments"]
                dim_rmse = baseline_estimation(d_data, upper_bound, lower_bound, e_prime, num_experiments)
                exp_err += dim_rmse * dim_rmse
            exp_err = np.sqrt(exp_err / len(dims))
            algo_err.append(exp_err)
    else:
        for e in epsilons:
            if config["concentration_algorithm"] == "quantiles":
                e_prime = e / dim_qtile
            else:
                e_prime = e / dim_qtile_base
            exp_err = 0
            for d in dims:
                print(d)
                d_data = data[data["Dimension"]==d]
                L = metadata[metadata["Dimension"]==d]["L"].values[0]
                user_arrays, K = get_user_arrays(d_data, L, config["user_groupping"])
                actual_mean = np.mean(d_data["Value"].values)
                user_group_means = [np.mean(x) for x in user_arrays]
                dim_rmse = private_estimation(
                    user_group_means,
                    L,
                    K,
                    config["data"][dataset]["upper_bound"],
                    config["data"][dataset]["lower_bound"],
                    e_prime,
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
    