import numpy as np
import cvxpy as cp

from tqdm import tqdm

def setup_cvx_problem(
    x_calib, scores_calib, alpha
):
    params = cp.Variable(name="params", shape=x_calib.shape[1])
    intercept = cp.Variable(name="intercept")

    x_const = cp.Constant(x_calib)
    x_param = cp.Parameter(name="x_test", shape=(1, x_calib.shape[1]))
    X = cp.vstack([x_const, x_param])

    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    scores_param = cp.Parameter(name="score_impute", shape=(1,1))
    scores = cp.vstack([scores_const, scores_param])

    resid = cp.vec(scores) - (X @ params) - intercept
    loss = cp.sum(0.5 * cp.abs(resid) + cp.multiply(alpha - 0.5, resid))

    prob = cp.Problem(cp.Minimize(loss))

    return prob

def compute_threshold(
    prob, x_test, scores_calib, M = None
):
    valid_points = []
    if M is None:
        m = np.max(scores_calib)
        threshold, _ = _compute_adaptive_threshold(prob, x_test, m)
        valid_points.append(threshold)
    elif M == "full":
        for m in tqdm(np.linspace(0, np.max(scores_calib), 100)):
            threshold, _ = _compute_adaptive_threshold(prob, x_test, m)
            if m < threshold:
                valid_points.append(m)
    else:
        threshold, _ = _compute_adaptive_threshold(prob, x_test, M)
        valid_points.append(threshold)

    return np.max(valid_points)

    
def _compute_adaptive_threshold(
    prob, x_test, M
):
    prob.param_dict["score_impute"].value = np.asarray([[M]])
    prob.param_dict["x_test"].value = x_test.reshape(1,-1)
    prob.solve(solver='MOSEK', verbose=False)
    
    var_dict = {}
    var_dict['params'] = prob.var_dict['params'].value
    var_dict['intercept'] = prob.var_dict['intercept'].value

    threshold = x_test.reshape(1,-1) @ var_dict['params'] + var_dict['intercept']

    return threshold, var_dict

def compute_group_coverages(
    x_calib, scores_calib, scores_test, groups_test, x_test, alpha
):
    coverages = {i : [] for i in range(groups_test.shape[1])}
    coverages[-1] = []

    prob = setup_cvx_problem(x_calib, scores_calib, alpha)

    for i in tqdm(range(len(x_test))):
        threshold, _ = _compute_adaptive_threshold(
            prob, x_test[i,:], np.max(scores_calib)
        )
        coverage = scores_test[i] < threshold
        for gp in np.where(groups_test[i] > 0)[0]:
            coverages[gp].append(coverage)
        coverages[-1].append(coverage)

    return {gp : np.mean(coverages) for gp, coverages in coverages.items()}

def compute_split_coverages(
    scores_calib, scores_test, groups_test, alpha
):
    split_threshold = np.quantile(scores_calib, [alpha * (1 + 1/len(scores_calib))])
    split_coverages = {i : [] for i in range(groups_test.shape[1])}
    split_coverages[-1] = []
    
    for i in tqdm(range(len(groups_test))):
        coverage = scores_test[i] < split_threshold
        for gp in np.where(groups_test[i] > 0)[0]:
            split_coverages[gp].append(coverage)
        split_coverages[-1].append(coverage)

    return {gp : np.mean(coverages) for gp, coverages in split_coverages.items()}