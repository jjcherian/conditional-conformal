import cvxpy as cp
import numpy as np

from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from conditionalconformal import CondConf
from conditionalconformal.synthetic_data import generate_cqr_data, indicator_matrix

# coverage on indicators of all sub-intervals with endpoints in [0,0.5,1,..,5]
eps = 0.5
disc = np.arange(0, 5 + eps, eps)

def phi_fn_groups(x):
    return indicator_matrix(x, disc)

# coverage on Gaussians with mu=loc and sd=scale 
# scale = 1 for x != [1.5, 3.5]
eval_locs = [1.5, 3.5]
eval_scale = 0.2

other_locs = [0.5, 2.5, 4.5]
other_scale = 1

def phi_fn_shifts(x):
    shifts = [norm.pdf(x, loc=loc, scale=eval_scale).reshape(-1,1)
                   for loc in eval_locs]
    shifts.extend([norm.pdf(x, loc=loc, scale=other_scale).reshape(-1,1)
                   for loc in other_locs])
    shifts.append(np.ones((x.shape[0], 1)))
    return np.concatenate(shifts, axis=1)

# intercept only phi_fn
def phi_fn_intercept(x):
    return np.ones((x.shape[0], 1))

def eval_shifts(x):
    shifts = [np.ones((x.shape[0], 1))]
    shifts.extend([norm.pdf(x, loc=loc, scale=eval_scale).reshape(-1,1)
                   for loc in eval_locs])
    return np.concatenate(shifts, axis=1)

def eval_groups(x):
    i_12 = ((x >= 1) & (x <= 2)).flatten()
    i_34 = ((x >= 3) & (x <= 4)).flatten()
    shifts = [np.ones((x.shape[0], 1)), i_12.reshape(-1,1), i_34.reshape(-1,1)]
    return np.concatenate(shifts, axis=1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="groups")
    parser.add_argument('-n', '--num_trials', default=200, type=int)
    return parser.parse_args()


def run_trial(method, seed, x_train, y_train):
    _, _, x_calib, y_calib, x_test, y_test = generate_cqr_data(seed=seed)

    alpha = 0.1

    # fit a fourth order polynomial
    poly = PolynomialFeatures(4)
    reg = LinearRegression().fit(poly.fit_transform(x_train), y_train)

    # score function is residual
    score_fn = lambda x, y : y - reg.predict(poly.fit_transform(x))

    if method == 'groups':
        phi_fn = phi_fn_groups  
        infinite_params = {}
    elif method == 'shifts':
        phi_fn = phi_fn_shifts
        infinite_params = {}
    elif method == 'agnostic':
        phi_fn = phi_fn_intercept
        infinite_params = {'kernel': 'rbf', 'gamma': 12.5, 'lambda': 0.005}

    cond_conf = CondConf(score_fn, phi_fn, infinite_params)
    cond_conf.setup_problem(x_calib, y_calib)

    cov_ub = cond_conf.verify_coverage(x_test, y_test, 1 - alpha/2)
    cov_lb = cond_conf.verify_coverage(x_test, y_test, alpha/2)

    cov_gcc = (~cov_ub | cov_lb).flatten()

    scores_test = np.abs(y_test - reg.predict(poly.fit_transform(x_test)))

    q = np.quantile(np.abs(reg.predict(poly.fit_transform(x_calib)) - y_calib),
                np.ceil((len(x_calib) + 1) * (0.9)) / len(x_calib),)
    
    cov_split = scores_test > q

    if method == 'groups':
        weights = eval_groups(x_test)
        weights /= np.sum(weights, axis=0)
    else:
        weights = eval_shifts(x_test)
        weights /= np.sum(weights, axis=0)

    cov_marg = [np.mean(cov_split), np.mean(cov_gcc)]
    cov_1 = [np.sum(weights[:,1] * cov_split), np.sum(weights[:,1] * cov_gcc)]
    cov_2 = [np.sum(weights[:,2] * cov_split), np.sum(weights[:,2] * cov_gcc)]
    return cov_marg, cov_1, cov_2

args = parse_args()

m = []
one = []
two = []
x_train, y_train, _, _, _, _ = generate_cqr_data(seed=1)

pbar = tqdm(total=args.num_trials, dynamic_ncols=True)
for trial in range(args.num_trials):
    try:
        marginal, g1, g2 = run_trial(method=args.method, seed=trial + 1, x_train=x_train, y_train=y_train)
    except cp.SolverError:
        marginal, g1, g2 = run_trial(method=args.method, seed=trial * 1000, x_train=x_train, y_train=y_train)
    m.append(marginal)
    one.append(g1)
    two.append(g2)

    pbar.set_description(f'Running averages: {np.mean(m, axis=0)}, {np.mean(one, axis=0)}, {np.mean(two, axis=0)}')
    
    pbar.update()

pbar.close()

import pickle

with open(f'data/{args.method}_results.pkl', 'wb') as fp:
    pickle.dump((m, one, two), fp)