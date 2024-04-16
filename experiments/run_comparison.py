import argparse
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from conditionalconformal.experiment_utils import run_experiment

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_trials', '-n', type=int, default=10)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('methods', nargs='+')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    attrib = pd.read_csv('data/CandCData/attributes.csv', delim_whitespace = True)
    data = pd.read_csv('data/CandCData/communities.data', names = attrib['attributes'])
    data['intercept'] = np.ones((len(data),))
    # drop all columns with missing values
    data.replace('?', pd.NA, inplace=True)  # Replace question marks with NaN
    data_cleaned = data.dropna(axis=1)

    orig_features = ['population','racepctblack','racePctWhite','racePctAsian',
                'racePctHisp','agePct12t21','agePct65up','medIncome','PctUnemployed','ViolentCrimesPerPop']

    # obtain all features except the metadata
    features = [c for c in data_cleaned.columns if c not in ['communityname', 'fold']]

    dataSub = data_cleaned[features]

    X = dataSub.drop(['ViolentCrimesPerPop'],axis=1).to_numpy()
    Y = dataSub['ViolentCrimesPerPop'].to_numpy()

    n_test = 694
    n_train = 650
    n_calib = 650
    n_trials = 200
    alpha = 0.1

    trial_results = []

    for seed in tqdm(range(args.seed, args.seed + args.n_trials)):
        # x_test, coverages = run_coverage_experiment((X, Y), n_test, n_calib, alpha, methods=args.methods, seed=seed)
        x_test, res = run_experiment((X, Y), n_test, n_calib, alpha, methods=args.methods, seed=seed)
        trial_results.append(
            {'x_test': x_test, 'length': res[0], 'coverage': res[1]}
        )
        if seed % 10 == 0:
            print(seed)
    
    with open(f'results_{args.seed}.pkl', 'wb') as fp:
        pickle.dump(trial_results, fp)
