import json
from utils import load_data, calc_dim_qtile, calc_min_k, get_max_k_dim
from utils import get_k_post_drop, drop_dim_for_user, calc_user_array_length
from groupping import get_user_arrays
from estimation import private_estimation, baseline_estimation

import numpy as np

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile) 
        print("Configurations loaded from config.json")
        jsonfile.close()
    dataset = config["dataset"]
    data_non_iid = load_data(dataset, config["data"][dataset])
    data = data_non_iid
    
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
    users = data["User"].unique()
    # for d in dims:
    #     d_data = data[(data["Dimension"] == d) & (data["User"] == users[0])]
    #     print(d_data)
    dim_qtile = calc_dim_qtile(data, 90)
    min_k = calc_min_k(data)
    for u in users:
        user_data = data[data["User"]==u]
        num_user_dims = len(user_data["Dimension"].unique())
        while num_user_dims > dim_qtile:
            user_data = data[data["User"]==u]
            max_k_dim = get_max_k_dim(user_data, data)
            k_post_drop = get_k_post_drop(data, max_k_dim, u)
            if k_post_drop < min_k:
                # dim_qtile = num_user_dims
                break
            else:
                num_user_dims -=1
                data = drop_dim_for_user(data, max_k_dim, u)
    print("After dropping dimensions: ")            
    dim_qtile = calc_dim_qtile(data, 85)
    print("After dropping dimensions: ")
    dim_qtile = calc_dim_qtile(data, 100)
    # L = calc_user_array_length(data, type=config["user_group_size"])
    # print("L: ", L)
    # user_arrays, K = get_user_arrays(data, L, config["user_groupping"])
    # print("K:", K)
    # actual_mean = np.mean(data["Value"].values)
    # user_group_means = [np.mean(x) for x in user_arrays]
    # np.save("./user_group_mean.npy", user_group_means)
    # epsilons = config["epsilons"]
    # upper_bound = config["data"][dataset]["upper_bound"]
    # lower_bound = config["data"][dataset]["lower_bound"]
    # num_experiments = config["num_experiments"]
    # print("Actual mean: ", actual_mean)
    # baseline_estimation(data, upper_bound, lower_bound, epsilons, num_experiments)
    # for e in epsilons:
    #     private_estimation(
    #         user_group_means,
    #         L,
    #         K,
    #         upper_bound,
    #         lower_bound,
    #         e,
    #         num_experiments,
    #         actual_mean,
    #         config["user_groupping"],
    #         config["concentration_algorithm"],
    #         config["algorithm_parameters"][config["concentration_algorithm"]]
    #     )