import json
import os
from utils import load_data, std_preproc_itms

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile)
        print("Configurations loaded from config.json")
        jsonfile.close()

    dataset = config["dataset"]
    data = std_preproc_itms(config["data"][dataset])
    data.to_csv("./data/ITMS_processed.csv", index=False)
    load_data(data, config["num_experiments"], config["data"][dataset])
