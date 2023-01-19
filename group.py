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
    x_calib, scores_calib, scores_test, groups_test, x_test, alpha, exact = False
):
    coverages = {i : [] for i in range(groups_test.shape[1])}
    coverages[-1] = []

    prob = setup_cvx_problem(x_calib, scores_calib, alpha)

    for i in tqdm(range(len(x_test))):
        if not exact:
            threshold, _ = _compute_adaptive_threshold(
                prob, x_test[i,:], np.max(scores_calib)
            )
        else:
            threshold, _ = _compute_adaptive_threshold(
                prob, x_test[i,:], scores_test[i]
            )
        coverage = scores_test[i] <= threshold
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
    
    for i in (range(len(groups_test))):
        coverage = scores_test[i] <= split_threshold
        for gp in np.where(groups_test[i] > 0)[0]:
            split_coverages[gp].append(coverage)
        split_coverages[-1].append(coverage)

    return {gp : np.mean(coverages) for gp, coverages in split_coverages.items()}

def compute_qr_coverages(
    groups_train, groups_test, scores_train, scores_test, alpha
    
):
    prob = setup_cvx_problem(groups_train[0:(len(scores_train)-1),:], scores_train[0:(len(scores_train)-1)], alpha)
    prob.param_dict["score_impute"].value = np.asarray([[scores_train[-1]]])
    prob.param_dict["x_test"].value = groups_train[-1,:].reshape(1,groups_train[-1,:].shape[0])
    prob.solve(solver='MOSEK', verbose=False)
    
    beta = prob.var_dict['params'].value
    beta0 = prob.var_dict['intercept'].value

    qr_coverages = {i : [] for i in range(groups_test.shape[1])}
    qr_coverages[-1] = []

    for i in (range(len(groups_test))):
        coverage = scores_test[i] <= beta0 + beta@groups_test[i,:] 
        for gp in np.where(groups_test[i] > 0)[0]:
            qr_coverages[gp].append(coverage)
        qr_coverages[-1].append(coverage)
        
    return {gp : np.mean(coverages) for gp, coverages in qr_coverages.items()}


def compute_cqr_coverages(
    groups_train, groups_calib, groups_test, scores_train, scores_calib, scores_test, alpha
):
    prob = setup_cvx_problem(groups_train[0:(len(scores_train)-1),:], scores_train[0:(len(scores_train)-1)], alpha)
    prob.param_dict["score_impute"].value = np.asarray([[scores_train[-1]]])
    prob.param_dict["x_test"].value = groups_train[-1,:].reshape(1,groups_train[-1,:].shape[0])
    prob.solve(solver='MOSEK', verbose=False)
    
    beta = prob.var_dict['params'].value
    beta0 = prob.var_dict['intercept'].value
    
    qPred = groups_calib@beta + beta0
    scores = scores_calib-qPred
    q = np.quantile(scores, [alpha * (1 + 1/len(scores_calib))])
    
    cqr_coverages = {i : [] for i in range(groups_test.shape[1])}
    cqr_coverages[-1] = []

    for i in (range(len(groups_test))):
        coverage = scores_test[i] <= beta0 + beta@groups_test[i,:] + q
        for gp in np.where(groups_test[i] > 0)[0]:
            cqr_coverages[gp].append(coverage)
        cqr_coverages[-1].append(coverage)
    return {gp : np.mean(coverages) for gp, coverages in cqr_coverages.items()}

# n = 1250
# x_std = 0.1
# y_std = 0.25
# d = 15 # choose d > 10

# # std_dev_list = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# std_dev_list = np.array([5.0, 0.1, 5.0, 0.1, 0.1, 5.0, 0.1, 5.0, 5.0, 0.1]) 

# theta = np.random.normal(loc=np.zeros(d), scale=x_std)
# from Synthetic_data_generation import generate_group_synthetic_data, get_groups
# x_train_final, y_train_final, x_calib, y_calib, x_test, y_test = generate_group_synthetic_data(
#     n, x_std, y_std, d, std_dev_list, theta, 100, 100
# )
# groups_test = get_groups(x_test[:,0:10])
# groups_calib = get_groups(x_calib[:,0:10])
# groups_train = get_groups(x_train_final[:,0:10])
# scores_train = y_train_final
# scores_calib = y_calib
# scores_test = y_test

# compute_group_coverages(
#         groups_train, y_train_final, y_test, groups_test, groups_test, 0.9, exact = False
#     )