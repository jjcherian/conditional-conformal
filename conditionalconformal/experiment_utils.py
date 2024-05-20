import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from quantile_forest import RandomForestQuantileRegressor

from conditionalconformal import CondConf

## get base model for constructing scores
def fit_model(data_train, base_model):
    x_train, y_train = data_train
    if base_model == "ols":
        reg = LinearRegression().fit(x_train, y_train)
    elif base_model == "qrf":
        reg = RandomForestQuantileRegressor()
        reg.fit(x_train, y_train)
    elif base_model == "qr":
        reg = CondConf(score_fn = lambda x, y: y, Phi_fn = lambda x: x)
        reg.setup_problem(x_train, y_train)
        # overwrite prediction function so it looks like a regression object
        reg.predict = lambda x, q: (x @ reg._get_calibration_solution(q)[1]).flatten() # expects x to be of form (n_points, n_feats)
    return reg

# helper function for splitting dataset
def split_dataset(dataset, n_test, n_calib, rng):
    X, Y = dataset
    data_indices = np.arange(len(X))
    rng.shuffle(data_indices)
    test_indices, calib_indices, train_indices = np.array_split(
        data_indices, 
        np.cumsum([n_test, n_calib])
    )

    X_test = X[test_indices]
    Y_test = Y[test_indices]

    X_calib = X[calib_indices]
    Y_calib = Y[calib_indices]

    X_train = X[train_indices]
    Y_train = Y[train_indices]
    return (X_train, Y_train), (X_calib, Y_calib), (X_test, Y_test)

# get coverages for each method type...
def get_coverage(dataset_calib, dataset_test, score_fn, method, quantile):
    if method == "split":
        scores_calib = score_fn(*dataset_calib)
        scores_test = score_fn(*dataset_test)

        score_cutoff = np.quantile(
            scores_calib, 
            [quantile * (1 + 1/len(scores_calib))]
        )
        if quantile >= 0.5:
            covs = scores_test <= score_cutoff
        else:
            covs = scores_test >= score_cutoff
    elif "rand" in method:
        condcalib = CondConf(score_fn, lambda x: x)
        condcalib.setup_problem(*dataset_calib)
        X_test, Y_test = dataset_test
        covs = condcalib.verify_coverage(X_test, Y_test, quantile, resolve=True, randomize=True)
    else:
        condcalib = CondConf(score_fn, lambda x: x)
        condcalib.setup_problem(*dataset_calib)
        X_test, Y_test = dataset_test
        covs = condcalib.verify_coverage(X_test, Y_test, quantile, resolve=True, randomize=False)
    return covs

# get coverages for each method type...
def get_cutoff(dataset_calib, dataset_test, score_fn, method, quantile):
    print(method, quantile)
    scores_test = score_fn(*dataset_test)
    if method == "split":
        scores_calib = score_fn(*dataset_calib)
        score_cutoff = np.quantile(
            scores_calib, 
            [quantile * (1 + 1/len(scores_calib))]
        )
        cutoffs = np.ones((len(scores_test,))) * score_cutoff
    elif "rand" in method:
        condcalib = CondConf(score_fn, lambda x: x)
        condcalib.setup_problem(*dataset_calib)
        cutoffs = []
        for x in dataset_test[0]:
            cutoff = condcalib.predict(quantile, x, lambda c, x: c, randomize=True)
            cutoffs.append(cutoff)
        cutoffs = np.asarray(cutoffs)
    else:
        condcalib = CondConf(score_fn, lambda x: x)
        condcalib.setup_problem(*dataset_calib)
        cutoffs = []
        for x in dataset_test[0]:
            cutoff = condcalib.predict(quantile, x, lambda c, x: c, randomize=False)
            cutoffs.append(cutoff)
        cutoffs = np.asarray(cutoffs)
    if quantile > 0.5:
        coverages = scores_test <= cutoffs.flatten()
    else:
        coverages = scores_test >= cutoffs.flatten()
    return cutoffs, coverages


def run_coverage_experiment(dataset, n_test, n_calib, alpha, methods = [], seed = 0):
    rng = np.random.default_rng(seed=seed)

    dataset_train, dataset_calib, dataset_test = split_dataset(
        dataset,
        n_test,
        n_calib,
        rng
    )

    ### Compute conformity scores
    base_methods = set([m.split('-')[0] for m in methods])
    base_model = {base : fit_model(dataset_train, base) for base in base_methods}
    
    coverages = []
    # example methods: (BASE_METHOD)-(CONFORMAL_METHOD)
    # BASE_METHOD valid choices: "ols", "qr", "qrf"
    # CONFORMAL_METHOD valid choices: "split", "cc", "ccrand", "lcp", "rlcp" (todo on last two)
    for method in methods:
        base_method, conformal_method = method.split('-')
        reg = base_model[base_method]
        if "q" in base_method: # if a quantile regression score needs to specify quantile
            score_fn_upper = lambda x, y: y - reg.predict(x, 1 - alpha/2)
            score_fn_lower = lambda x, y: y - reg.predict(x, alpha/2)
        else:
            score_fn_upper = lambda x, y: y - reg.predict(x)
            score_fn_lower = lambda x, y: y - reg.predict(x)
        covers_upper = get_coverage(dataset_calib, dataset_test, score_fn_upper, conformal_method, 1 - alpha/2)
        covers_lower = get_coverage(dataset_calib, dataset_test, score_fn_lower, conformal_method, alpha/2)
        covers = np.logical_and(covers_upper, covers_lower)
        coverages.append(covers)

    return dataset_test[0], coverages


def run_experiment(dataset, n_test, n_calib, alpha, methods = [], seed = 0):
    rng = np.random.default_rng(seed=seed)

    dataset_train, dataset_calib, dataset_test = split_dataset(
        dataset,
        n_test,
        n_calib,
        rng
    )

    ### Compute conformity scores
    base_model = {base : fit_model(dataset_train, base) for base in ["ols", "qrf", "qr"]}
    
    all_lengths = []
    all_coverages = []
    # example methods: (BASE_METHOD)-(CONFORMAL_METHOD)
    # BASE_METHOD valid choices: "ols", "qr", "qrf"
    # CONFORMAL_METHOD valid choices: "split", "cc", "ccrand", "lcp", "ccqp"
    for method in methods:
        base_method, conformal_method = method.split('-')
        reg = base_model[base_method]
        if "qrf" in base_method: # if a quantile regression score needs to specify quantile
            score_fn_upper = lambda x, y: y - reg.predict(x, 1 - alpha/2) + rng.uniform(0, 1e-5, size=len(x))
            score_fn_lower = lambda x, y: y - reg.predict(x, alpha/2) + rng.uniform(0, 1e-5, size=len(x))
        elif "q" in base_method:
            score_fn_upper = lambda x, y: y - reg.predict(x, 1 - alpha/2)
            score_fn_lower = lambda x, y: y - reg.predict(x, alpha/2)
        else:
            score_fn_upper = lambda x, y: y - reg.predict(x)
            score_fn_lower = lambda x, y: y - reg.predict(x)
        cutoffs_upper, cov_upper = get_cutoff(dataset_calib, dataset_test, score_fn_upper, conformal_method, 1 - alpha/2)
        cutoffs_lower, cov_lower = get_cutoff(dataset_calib, dataset_test, score_fn_lower, conformal_method, alpha/2)
        if "q" in base_method:
            pred_upper = reg.predict(dataset_test[0], 1 - alpha/2)
            pred_lower = reg.predict(dataset_test[0], alpha/2)
            pred_gap = pred_upper - pred_lower
        else:
            pred_gap = 0
        lengths = cutoffs_upper - cutoffs_lower + pred_gap
        coverage = np.logical_and(cov_upper, cov_lower)
        all_lengths.append(lengths)
        all_coverages.append(coverage)

    return dataset_test[0], (all_lengths, all_coverages)