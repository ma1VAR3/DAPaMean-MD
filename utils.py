import json
import math
import numpy as np
import pandas as pd
import h3

def load_data(dataset="ITMS", config=None):
    data = None
        
    if dataset == "ITMS":
        df = pd.read_csv("./suratITMSDPtest/suratITMSDPtest.csv")
        df = df.drop_duplicates(subset=['trip_id', 'observationDateTime'], ignore_index=True)
        df = df.drop(columns = [
                    "trip_direction",
                    "last_stop_id",
                    "last_stop_arrival_time",
                    "route_id",
                    "actual_trip_start_time",
                    "trip_delay",
                    "vehicle_label",
                    "id",
                    "location.type",
                    "trip_id"
                ])
        lat_lon = df["location.coordinates"].astype(str).str.strip('[]').str.split(",")
        lon = lat_lon.apply(lambda x: x[0])
        lat = lat_lon.apply(lambda x: x[1])
        dflen = len(df)
        h3index = [None] * dflen
        resolution = config["H3 Resolution"]
        for i in range(dflen):
            h3index[i] = h3.geo_to_h3(lat=float(lat[i]), lng=float(lon[i]), resolution=resolution)
        df["h3index"] = h3index
        df["Date"] = pd.to_datetime(df["observationDateTime"]).dt.date
        df["Time"] = pd.to_datetime(df["observationDateTime"]).dt.time
        time = df["Time"]
        df["Timeslot"] = time.apply(lambda x: x.hour)
        df["HAT"] = (df["Timeslot"].astype(str) + " " + df["h3index"])
        startTime = config["Start time"]
        endTime = config["End time"]
        df = df[(df["Timeslot"] >= startTime) & (df["Timeslot"] <= endTime)]
        df = df[df["speed"]>0]
        h_d = df
        h_d = h_d.drop(columns = [
            "observationDateTime",
            "location.coordinates",
            "h3index",
            "Date",
            "Time",
            "Timeslot",
        ])
        h_d = h_d.rename(columns = {
            'license_plate':'User',
            'speed' : 'Value',
            'HAT' : 'Dimension'
        })
        data = h_d
        
        dims = data["Dimension"].unique()
        users = data["User"].unique()
        l_vals = []
        k_vals = []
        for d in dims:
            dim_data = data[data["Dimension"]==d]
            dim_l = calc_user_array_length(dim_data)
            l_vals.append(dim_l)
            dim_k = calc_k(dim_data, dim_l)
            # dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
            # dim_k = np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l
            k_vals.append(dim_k)
        metadata_dict = {
            "Dimension" : dims,
            "L" : l_vals,
            "K" : k_vals
        }
        metadata_df = pd.DataFrame(metadata_dict)
        filtered_metadata_df = metadata_df[metadata_df["K"]>20]
        filtered_dims = filtered_metadata_df["Dimension"].unique()
        filtered_data = data[data["Dimension"].isin(filtered_dims)]
        
    return filtered_data, filtered_metadata_df

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

def get_k_ordered_dims(user_data, metadata):
    """
    Get the k ordered dimensions for a user
    """
    user_dims = user_data["Dimension"].unique()
    user_metadata = metadata[metadata["Dimension"].isin(user_dims)]
    user_metadata = user_metadata.sort_values(by=["K"], ascending=False)
    k_ordered_dims = user_metadata["Dimension"].values
    return k_ordered_dims

def calc_dim_qtile_dropping(data, metadata, users, support, qtile):
    """
    The two pass approach to calculate threhold for dropping dimensions
    """
    dim_qtile = calc_dim_qtile(data, qtile) # <- Initial threshold
    for u in users:
        user_data = data[data["User"]==u]
        num_user_dims = len(user_data["Dimension"].unique())
        flag = False # <- Flag to break out of loop of dropping dims
        while num_user_dims > dim_qtile and not flag:
            user_data = data[data["User"]==u] # <- Get user data again (since dims have been dropped)
            k_ordered_dims = get_k_ordered_dims(user_data, metadata)
            for s_dim in k_ordered_dims:
                if metadata[metadata["Dimension"]==s_dim]["K"].values[0] < support:
                    break
                elif metadata[metadata["Dimension"]==s_dim]["K"].values[0]-1 < support:
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
                    
    dim_qtile = calc_dim_qtile(data, 100) # <- Final threshold
    return dim_qtile

def calc_support(metadata):
    
    k_vals = metadata["K"].values
    
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
    
    return np.min(k_vals)

def get_max_k_dim(user_data, data):
    """
    Gets the dimension with the maximum k value for the given user data
    """
    user_dims = user_data["Dimension"].unique()
    max_user_k_val = -1
    max_k_val_dim = None
    for d in user_dims:
        dim_data = data[data["Dimension"]==d]
        dim_l = calc_user_array_length(dim_data)
        dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
        dim_k = np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l
        if dim_k > max_user_k_val:
            max_user_k_val = dim_k
            max_k_val_dim = d
    return max_k_val_dim

def get_support_post_drop(data, dim, user):
    """
    Gets the k value for the given dimension after dropping the dimension for the given user
    """
    new_data = drop_dim_for_user(data, dim, user)
    dim_data = new_data[new_data["Dimension"]==dim]
    dim_l = calc_user_array_length(dim_data)
    dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    dim_k = np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l
    return dim_k

def drop_dim_for_user(data, metadata, dim, user):
    """
    Drops the rows corresponding to the given dimension for the given user
    """
    
    entries_to_drop = data[(data["Dimension"]==dim) & (data["User"]==user)]
    new_data = data.drop(entries_to_drop.index, inplace=False)
    new_k_for_dim = calc_k(new_data[new_data["Dimension"]==dim], metadata[metadata["Dimension"]==dim]["L"].values[0])
    metadata.loc[metadata["Dimension"]==dim, "K"] = new_k_for_dim
    return new_data, metadata

def calc_user_array_length(data, type="median"):
    L = None
    data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    if type=="median":
        L = math.floor(np.median(data_grouped["Value"]))
    elif type=="mean":
        L = math.floor(np.mean(data_grouped["Value"]))
    elif type=="max":
        L = np.max(data_grouped["Value"])
    elif type=="rms":
        L = math.floor(math.sqrt(np.mean([math.pow(i, 2) for i in data_grouped["Value"]])))
    
    return L


