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
    return data

def dim_qtile(data):
    dims = data["Dimension"].unique()
    print("ALL DIMS: ", len(dims))
    users = data["User"].unique()
    dims = []
    for u in users:
        user_data = data[data["User"]==u]
        dims.append(len(user_data["Dimension"].unique()))
        
    # import plotly.express as px
    # fig = px.histogram(dims)
    # fig.show()
    
    dim_qtile = np.percentile(dims, 90)
    print("Max dim contributed: ", np.max(dims))
    print("90th percentile of dims contributed: ", dim_qtile)
    return dim_qtile

def calc_min_k(data):
    dims = data["Dimension"].unique()
    users = data["User"].unique()
    k_vals = []
    for d in dims:
        dim_data = data[data["Dimension"]==d]
        dim_l = calc_user_array_length(dim_data)
        dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
        dim_k = np.floor(np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l)
        k_vals.append(dim_k)
        
    # import plotly.express as px
    # fig = px.histogram(k_vals)
    # fig.show()
    
    print("Min k: ", np.min(k_vals))
    
    return np.min(k_vals)

def get_max_k_dim(user_data, data):
    user_dims = user_data["Dimension"].unique()
    max_user_k_val = -1
    max_k_val_dim = None
    for d in user_dims:
        dim_data = data[data["Dimension"]==d]
        dim_l = calc_user_array_length(dim_data)
        dim_data_grouped = dim_data.groupby(["User"]).agg({"Value": "count"}).reset_index()
        dim_k = np.floor(np.sum([np.minimum(i, dim_l) for i in dim_data_grouped["Value"]]) / dim_l)
        if dim_k > max_user_k_val:
            max_user_k_val = dim_k
            max_k_val_dim = d
    return max_k_val_dim

def get_k_post_drop(data, dim, user):
    pass

def drop_dim_for_user(data, dim, user):
    pass

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