import json
import os
from utils import load_data

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile)
        print("Configurations loaded from config.json")
        jsonfile.close()

    dataset = config["dataset"]
    data, metadata = load_data(dataset, config["data"][dataset])
    os.makedirs("./data", exist_ok=True)
    if config["data"][dataset]["Synthetic"] == False:
        data.to_csv("./data/filtered_data.csv", index=False)
        metadata.to_csv("./data/filtered_metadata.csv", index=False)
    else:
        data.to_csv("./data/synthetic_data.csv", index=False)
        metadata.to_csv("./data/synthetic_metadata.csv", index=False)
