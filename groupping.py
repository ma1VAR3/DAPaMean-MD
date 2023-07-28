import numpy as np


# def create_array_template(data, L, exp_type):
#     user_arrays = {}
#     K = None
#     if exp_type == "wrap":
#         users = np.unique(data["User"])
#         data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
#         K = np.floor(np.sum([np.minimum(i, L) for i in data_grouped["Value"]]) / L)
#         # user_arrays = np.zeros((int(K), int(L)))
#         arr_idx = 0
#         val_idx = 0
#         for u in users:
#             counter = 0
#             user_samples_in_array = 0
#             user_data = data[data["User"] == u]["Value"].values
#             stop_cond = np.minimum(len(user_data), L)
#             user_dict = {}
#             while counter < stop_cond:
#                 # user_arrays[arr_idx][val_idx] = u
#                 counter += 1
#                 user_samples_in_array += 1
#                 user_dict[arr_idx] = user_samples_in_array
#                 val_idx += 1
#                 if val_idx >= L:
#                     val_idx = 0
#                     arr_idx += 1
#                     user_samples_in_array = 0
#                 if arr_idx >= K:
#                     break
#             user_arrays[u] = user_dict
#             if arr_idx >= K:
#                 break
#     elif exp_type == "best_fit":
#         users = np.unique(data["User"])
#         user_arrays_dict = {}
#         user_arrays = [[]]
#         for u in users:
#             counter = 0
#             user_data = data[data["User"] == u]["Value"].values
#             stop_cond = np.minimum(len(user_data), L)
#             remaining_spaces = [
#                 L - len(user_arrays[i]) - stop_cond for i in range(len(user_arrays))
#             ]
#             remaining_spaces = np.array(remaining_spaces)
#             remaining_spaces = np.where(remaining_spaces < 0, L, remaining_spaces)
#             array_idx_to_fill = None
#             if np.min(remaining_spaces) >= L:
#                 user_arrays.append([])
#                 array_idx_to_fill = -1
#             else:
#                 array_idx_to_fill = np.argmin(remaining_spaces)

#             while counter < stop_cond:
#                 user_arrays[array_idx_to_fill].append(user_data[counter])
#                 counter += 1
#             user_dict = {}
#             # while counter < stop_cond:
#             # user_arrays[array_idx_to_fill].append(u)
#             # counter += 1
#             user_dict[array_idx_to_fill] = stop_cond
#             user_arrays_dict[u] = user_dict

#         K = len(user_arrays)
#         user_arrays = user_arrays_dict
#     # print(user_arrays)
#     actual_user_list = np.unique(data["User"])
#     array_user_list = user_arrays.keys()
#     left_out_users = []
#     for u in actual_user_list:
#         if u not in array_user_list:
#             left_out_users.append(u)
#     print(left_out_users)
#     # print(user_arrays)
#     return user_arrays, K


# def get_user_arrays_new(data, L, K, array_template, exp_type):
#     user_arrays = None

#     if exp_type == "wrap":
#         user_arrays = [np.array([]) for i in range(int(K.item()))]
#         for u in array_template.keys():
#             user_meta = array_template[u]
#             user_data = data[data["User"] == u]["Value"].values
#             filled_samples = 0
#             for arr_no in user_meta.keys():
#                 # print(arr_no)
#                 user_arrays[arr_no] = np.append(
#                     user_arrays[arr_no], user_data[filled_samples : user_meta[arr_no]]
#                 )
#                 filled_samples += user_meta[arr_no]

#     # elif exp_type == "best_fit":
#     #     users = np.unique(data["User"])
#     #     user_arrays = [[]]
#     #     for u in users:
#     #         counter = 0
#     #         user_data = data[data["User"] == u]["Value"].values
#     #         stop_cond = np.minimum(len(user_data), L)
#     #         remaining_spaces = [
#     #             L - len(user_arrays[i]) - stop_cond for i in range(len(user_arrays))
#     #         ]
#     #         remaining_spaces = np.array(remaining_spaces)
#     #         remaining_spaces = np.where(remaining_spaces < 0, L, remaining_spaces)
#     #         array_idx_to_fill = None
#     #         if np.min(remaining_spaces) >= L:
#     #             user_arrays.append([])
#     #             array_idx_to_fill = -1
#     #         else:
#     #             array_idx_to_fill = np.argmin(remaining_spaces)

#     #         while counter < stop_cond:
#     #             user_arrays[array_idx_to_fill].append(user_data[counter])
#     #             counter += 1

#     #     K = len(user_arrays)
#     created_arr_len = len(user_arrays)
#     print("created len: ", created_arr_len)
#     print("Original: ", K)

#     return user_arrays


def get_weighted_mean_template(data, L, exp_type):
    array_meta = {}
    final_k = None
    users = np.unique(data["User"])
    data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    if exp_type == "wrap":
        K = np.floor(np.sum([np.minimum(i, L) for i in data_grouped["Value"]]) / L)
        array_weights = [0]
        curr_array = 0
        for u in users:
            user_dict = {}
            user_data = data[data["User"] == u]["Value"].values
            # user_mean = np.mean(user_data)
            total_weight = np.minimum(len(user_data), L)
            remaining_weight_in_curr_array = L - array_weights[curr_array]
            if remaining_weight_in_curr_array >= total_weight:
                user_dict[curr_array] = total_weight
                array_weights[curr_array] = array_weights[curr_array] + total_weight
                if array_weights[curr_array] == L:
                    if curr_array + 1 >= K:
                        array_meta[u] = user_dict
                        break
                    curr_array += 1
                    array_weights.append(0)
            elif remaining_weight_in_curr_array < total_weight:
                user_dict[curr_array] = remaining_weight_in_curr_array
                array_weights[curr_array] = L
                if curr_array + 1 >= K:
                    array_meta[u] = user_dict
                    break
                curr_array += 1
                array_weights.append(total_weight - remaining_weight_in_curr_array)
                user_dict[curr_array] = array_weights[curr_array]
            array_meta[u] = user_dict
        final_k = len(array_weights)
        # print(array_weights)

    if exp_type == "best_fit":
        array_weights = [0]
        curr_array = 0
        for u in users:
            user_dict = {}
            user_data = data[data["User"] == u]["Value"].values
            # user_mean = np.mean(user_data)
            total_weight = np.minimum(len(user_data), L)
            remaining_spaces = [
                L - array_weights[i] - total_weight for i in range(len(array_weights))
            ]
            remaining_spaces = np.array(remaining_spaces)
            remaining_spaces = np.where(remaining_spaces < 0, L, remaining_spaces)
            array_idx_to_fill = None

            if np.min(remaining_spaces) >= L:
                array_weights.append(0)
                array_idx_to_fill = len(array_weights) - 1
            else:
                array_idx_to_fill = np.argmin(remaining_spaces)

            array_weights[array_idx_to_fill] = (
                array_weights[array_idx_to_fill] + total_weight
            )
            user_dict[array_idx_to_fill] = total_weight
            array_meta[u] = user_dict
        final_k = len(array_weights)

        # print(array_weights)

    # actual_user_list = np.unique(data["User"])
    # array_user_list = array_meta.keys()
    # left_out_users = []
    # for u in actual_user_list:
    #     if u not in array_user_list:
    #         left_out_users.append(u)
    # print("Left out:", left_out_users)

    return array_meta, final_k


def get_array_mean_from_template(array_template, K, data):
    array_weights = np.zeros(K)
    array_sums = np.zeros(K)

    for u in array_template.keys():
        user_data_orig = data[data["User"] == u]["Value"].values
        user_data_mean = np.mean(user_data_orig)
        # user_data = [user_data_mean for i in range(len(user_data_orig))]
        user_dict = array_template[u]
        filled_samples = 0
        for arr_no in user_dict.keys():
            array_sums[arr_no] += user_data_mean * user_dict[arr_no]
            filled_samples += user_dict[arr_no]
            array_weights[arr_no] += user_dict[arr_no]

    # print("ARRAY WEIGTHS AFTER FORMATION: ", array_weights)
    return np.divide(array_sums, array_weights)


def get_user_arrays(data, L, exp_type):
    user_arrays = None
    K = None
    if exp_type == "wrap":
        users = np.unique(data["User"])
        data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
        K = np.floor(np.sum([np.minimum(i, L) for i in data_grouped["Value"]]) / L)
        user_arrays = np.zeros((int(K), int(L)))
        arr_idx = 0
        val_idx = 0
        for u in users:
            counter = 0
            user_data_orig = data[data["User"] == u]["Value"].values
            user_data_mean = np.mean(user_data_orig)
            user_data = [user_data_mean for i in range(len(user_data_orig))]
            stop_cond = np.minimum(len(user_data), L)
            while counter < stop_cond:
                user_arrays[arr_idx][val_idx] = user_data[counter]
                counter += 1
                val_idx += 1
                if val_idx >= L:
                    val_idx = 0
                    arr_idx += 1
                if arr_idx >= K:
                    break
            if arr_idx >= K:
                break

    elif exp_type == "best_fit":
        users = np.unique(data["User"])
        user_arrays = [[]]
        for u in users:
            counter = 0
            user_data_orig = data[data["User"] == u]["Value"].values
            user_data_mean = np.mean(user_data_orig)
            user_data = [user_data_mean for i in range(len(user_data_orig))]
            stop_cond = np.minimum(len(user_data), L)
            remaining_spaces = [
                L - len(user_arrays[i]) - stop_cond for i in range(len(user_arrays))
            ]
            remaining_spaces = np.array(remaining_spaces)
            remaining_spaces = np.where(remaining_spaces < 0, L, remaining_spaces)
            array_idx_to_fill = None
            if np.min(remaining_spaces) >= L:
                user_arrays.append([])
                array_idx_to_fill = -1
            else:
                array_idx_to_fill = np.argmin(remaining_spaces)

            while counter < stop_cond:
                user_arrays[array_idx_to_fill].append(user_data[counter])
                counter += 1

        K = len(user_arrays)

    return user_arrays, K


# def get_user_arrays(data, L, exp_type):
#     user_arrays = None
#     K = None
#     if exp_type == "wrap":
#         users = np.unique(data["User"])
#         data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
#         K = np.floor(np.sum([np.minimum(i, L) for i in data_grouped["Value"]]) / L)
#         user_arrays = np.zeros((int(K), int(L)))
#         arr_idx = 0
#         val_idx = 0
#         for u in users:
#             counter = 0
#             user_data = data[data["User"] == u]["Value"].values
#             stop_cond = np.minimum(len(user_data), L)
#             while counter < stop_cond:
#                 user_arrays[arr_idx][val_idx] = user_data[counter]
#                 counter += 1
#                 val_idx += 1
#                 if val_idx >= L:
#                     val_idx = 0
#                     arr_idx += 1
#                 if arr_idx >= K:
#                     break
#             if arr_idx >= K:
#                 break

#     elif exp_type == "best_fit":
#         users = np.unique(data["User"])
#         user_arrays = [[]]
#         for u in users:
#             counter = 0
#             user_data = data[data["User"] == u]["Value"].values
#             stop_cond = np.minimum(len(user_data), L)
#             remaining_spaces = [
#                 L - len(user_arrays[i]) - stop_cond for i in range(len(user_arrays))
#             ]
#             remaining_spaces = np.array(remaining_spaces)
#             remaining_spaces = np.where(remaining_spaces < 0, L, remaining_spaces)
#             array_idx_to_fill = None
#             if np.min(remaining_spaces) >= L:
#                 user_arrays.append([])
#                 array_idx_to_fill = -1
#             else:
#                 array_idx_to_fill = np.argmin(remaining_spaces)

#             while counter < stop_cond:
#                 user_arrays[array_idx_to_fill].append(user_data[counter])
#                 counter += 1

#         K = len(user_arrays)

#     return user_arrays, K
