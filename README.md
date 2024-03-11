# DAPaMean

The paper on this research is live on arXiv: https://arxiv.org/abs/2401.15906

### Docs

This repository contains code to run experiments for different setting available for the framework DAPaMean (single dimension).

The following files are crucial for running:

- config.json : Contains the configurations for the experiments (**make changes to this file only**)

- main.py : Contains the main pipeline, reads configurations from the config.json

- filter_data.py : Contains script to generate and store synthetic dataset in the gen_data directory

### Steps:

1. Install all the requirements (requirements.txt)
2. Place the dataset in data/
3. Make necesarry changes in config.json
4. Run filter_data.py
5. Run main.py

NOTE: There are 3 possible scenarios (original data, synthetic data in which number of samples are scaled, synthetic data in which number of users are scaled). You need to run the filter_data.py only ONCE for EACH scenario. For a given scenario, to execute different experiments (6 possible experiments), simply change settings in the config.json and run main.py.

### Possible experiments

| concentration_algorithm 	| user_groupping 	|
|-------------------------	|----------------	|
| baseline2               	| wrap           	|
| baseline2               	| best_fit       	|
| coarse_mean             	| wrap           	|
| coarse_mean             	| best_fit       	|
| quantiles               	| wrap           	|
| quantiles               	| best_fit       	|


### The config.json file

The following parameters need to be taken care of before running the experiments:

- Synthetic : true/false (If true then synthetic data is used. Note that you should have generated the data using filter_data.py)

- Synthetic Factor : samples/users (Relevant only when "Synthetic" field is true. "samples" generates synthetic data in which number of samples are scaled. "users" generated synthetic data in which number of users are scaled.)

- Synthetic Factor : A number (The factor by which the number of samples are scaled. Should be kept the same for all the experiments for homogeneity)

- epsilons : A list of numbers (The epsilons to be tested. Must be the same for all experiments)

- num_experiments : A number (The number of times each expeirment is conducted. Myst be the same for all experiments)

- user_grouppung : wrap/best_fit (The grouping algo to be used)

- concentration_algorithm : baseline2/coarse_mean/quantiles (The estimation algorithm to be used)
