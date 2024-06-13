#!/usr/bin/env python3.9

import argparse
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from conditionalconformal.experiment_utils import run_coverage_experiment, run_experiment

import ray

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_trials', '-n', type=int, default=10)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--type', '-t', default='coverage')
    parser.add_argument('methods', nargs='+')

    return parser.parse_args()

@ray.remote
def parallel_experiment(*args, **kwargs):
    return run_experiment(*args, **kwargs)

@ray.remote
def parallel_coverage_experiment(*args, **kwargs):
    return run_coverage_experiment(*args, **kwargs)

if __name__ == "__main__":
    args = parse_args()

    attrib = pd.read_csv('experiments/data/CandCData/attributes.csv', delim_whitespace = True)
    data = pd.read_csv('experiments/data/CandCData/communities.data', names = attrib['attributes'])
    data['intercept'] = np.ones((len(data),))
    # drop all columns with missing values
    data.replace('?', pd.NA, inplace=True)  # Replace question marks with NaN
    data_cleaned = data.dropna(axis=1)

    orig_features = ['intercept', 'population','racepctblack','racePctWhite','racePctAsian',
                'racePctHisp','agePct12t21','agePct65up','medIncome','PctUnemployed','ViolentCrimesPerPop']

    # obtain all features except the metadata
    features = [c for c in data_cleaned.columns if c not in ['communityname', 'fold']]

    dataSub = data_cleaned[orig_features]

    X = dataSub.drop(['ViolentCrimesPerPop'],axis=1).to_numpy()
    Y = dataSub['ViolentCrimesPerPop'].to_numpy()

    n_test = 694
    n_train = 650
    n_calib = 650
    n_trials = 200
    alpha = 0.1

    # connect to cluster
    ray.init(address="auto")

    results = []

    for seed in range(args.seed, args.seed + args.n_trials):
        if args.type == 'coverage':
            result = parallel_coverage_experiment.remote(
                (X, Y), n_test, n_calib, alpha, methods=args.methods, seed=seed
            )
        else:
            result = parallel_experiment.remote(
                (X, Y), n_test, n_calib, alpha, methods=args.methods, seed=seed
            )
        results.append(result)
    
    trial_results = ray.get(results)

    if args.type == 'all':
        trial_results = [{'x_test': r[0], 'length': r[1][0], 'coverage': r[1][1]} for r in trial_results]
    else:
        trial_results = [{'x_test': r[0], 'coverage': r[1]} for r in trial_results]
    
    with open(f'results_{args.type}.pkl', 'wb') as fp:
        pickle.dump(trial_results, fp)
