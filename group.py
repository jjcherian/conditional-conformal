import numpy as np
import cvxpy as cp

from tqdm import tqdm

def compute_adaptive_coverages(
    group_calib, scores_calib, scores_test, groups_test, x_test, alpha
):
    coverages = {i : [] for i in range(group_calib.shape[1])}
    
    def get_threshold(group_calib, scores_calib, x_test, group_test, M = None):
        if M is None:
            M = np.max(scores_calib)
        group_fit.value = np.concatenate([group_calib, group_test.reshape(1,-1)], axis=0)
        scores_fit.value = np.concatenate([scores_calib, [M]])
        prob.solve(solver='MOSEK')

        threshold = group_test @ g_params.value + intercept.value
        return threshold

    g_params = cp.Variable(group_calib.shape[1])
    intercept = cp.Variable()

    group_fit = cp.Parameter(shape=(group_calib.shape[0] + 1, group_calib.shape[1]))
    scores_fit = cp.Parameter(shape=(scores_calib.shape[0] + 1,))
    resid = scores_fit - (group_fit @ g_params) - intercept
    loss = cp.sum(0.5 * cp.abs(resid) + cp.multiply(alpha - 0.5, resid))

    prob = cp.Problem(cp.Minimize(loss))

    for i in tqdm(range(len(x_test))):
        threshold = get_threshold(group_calib, scores_calib, x_test[i,:], groups_test[i,:])
        coverage = scores_test[i] < threshold
        for gp in np.where(groups_test[i] > 0)[0]:
            coverages[gp].append(coverage)

    return {gp : np.mean(coverages) for gp, coverages in coverages.items()}

def compute_split_coverages(
    group_calib, scores_calib, scores_test, groups_test, alpha
):
    split_threshold = np.quantile(scores_calib, [alpha * (1 + 1/len(scores_calib))])
    split_coverages = {i : [] for i in range(group_calib.shape[1])}
    
    for i in tqdm(range(len(groups_test))):
        coverage = scores_test[i] < split_threshold
        for gp in np.where(groups_test[i] > 0)[0]:
            split_coverages[gp].append(coverage)

    return {gp : np.mean(coverages) for gp, coverages in split_coverages.items()}