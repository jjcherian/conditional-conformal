import numpy as np
import cvxpy as cp
import mosek

from tqdm import tqdm

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold

def setup_cvx_primal(
    x_calib, scores_calib, kernel, gamma, alpha, radius
):    
    n_calib = len(scores_calib)

    rkhs_weight = cp.Variable(name="weights", shape=n_calib + 1)
    intercept = cp.Variable(name="intercept")

    K_11, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)

    K_11 = cp.Constant(K_11)
    K_21 = cp.Parameter(name="K_12", shape=(n_calib, 1))
    K_12 = cp.Parameter(name="K_21", shape=(1, n_calib + 1))
    K = cp.hstack([K_11, K_21])
    K = cp.vstack([K, K_12])

    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    scores_param = cp.Parameter(name="score_impute", shape=(1,1))
    scores = cp.vstack([scores_const, scores_param])

    L_11_const = cp.Constant(
        np.hstack([L_11, np.zeros((L_11.shape[0], 1))])
    )
    L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
    L = cp.vstack([L_11_const, L_21_22_param])

    resid = cp.vec(scores) - (K @ rkhs_weight) - intercept
    loss = cp.sum(0.5 * cp.abs(resid) + cp.multiply(alpha - 0.5, resid)) * (1/(n_calib + 1))
    loss += (1/(2*radius)) * cp.sum_squares(L.T @ rkhs_weight)
    prob = cp.Problem(cp.Minimize(loss))
  
    return prob

def setup_full_cvx_dual(
    x_calib, scores_calib, kernel, gamma, alpha
):
    n_calib = len(scores_calib)

    rkhs_weight = cp.Variable(name="weights", shape=n_calib )

    _, L = _get_kernel_matrix(x_calib, kernel, gamma)

    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    
    lamb = cp.Parameter(name="lambda", shape=(1,1))

    L_const = cp.Constant(L)

    C = lamb *  (n_calib)

    constraints = [
        C * (alpha - 1) <= rkhs_weight,
        C * alpha >= rkhs_weight,
        cp.sum(cp.multiply(rkhs_weight, np.ones(n_calib))) == 0]
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(L_const.T @ rkhs_weight) -
                    cp.sum(cp.multiply(rkhs_weight, cp.vec(scores_const)))),
        constraints
    )

    return prob


def setup_cvx_dual(
    x_calib, scores_calib, kernel, gamma, alpha
):
    n_calib = len(scores_calib)

    rkhs_weight = cp.Variable(name="weights", shape=n_calib + 1)

    _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)

    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    scores_param = cp.Parameter(name="score_impute", shape=(1,1))
    scores = cp.vstack([scores_const, scores_param])
    
    lamb = cp.Parameter(name="lambda", shape=(1,1))

    L_11_const = cp.Constant(
        np.hstack([L_11, np.zeros((L_11.shape[0], 1))])
    )
    L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
    L = cp.vstack([L_11_const, L_21_22_param])

    C = lamb *  (n_calib + 1)

    constraints = [
        C * (alpha - 1) <= rkhs_weight,
        C * alpha >= rkhs_weight,
        cp.sum(cp.multiply(rkhs_weight, np.ones((n_calib + 1,)))) == 0]
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(L.T @ rkhs_weight) - cp.sum(cp.multiply(rkhs_weight, cp.vec(scores)))),
        constraints
    )

    return prob

def _get_kernel_matrix(x_calib, kernel, gamma):

    K = pairwise_kernels(
        X=x_calib,
        metric=kernel,
        gamma=gamma
    ) + 1e-5 * np.eye(len(x_calib))

    K_chol = np.linalg.cholesky(K)
    return K, K_chol

def finish_dual_setup(
    prob,
    scores_calib : np.ndarray, 
    x_calib : np.ndarray,
    x_test : np.ndarray, 
    kernel : str, 
    gamma : float = 1,
    M : float = None,
    radius : float = 1
):
    if M is None:
        M = np.max(scores_calib)
    prob.param_dict['score_impute'].value = np.asarray([[M]])

    K_12 = pairwise_kernels(
        X=np.concatenate([x_calib, x_test.reshape(1,-1)], axis=0),
        Y=x_test.reshape(1,-1),
        metric=kernel,
        gamma=gamma
    )

    if 'K_12' in prob.param_dict:
        prob.param_dict['K_12'].value = K_12[:-1]
        prob.param_dict['K_21'].value = K_12.T

    _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
    K_22 = pairwise_kernels(
        X=x_test.reshape(-1,1),
        metric=kernel,
        gamma=gamma
    )

    L_21 = np.linalg.solve(L_11, K_12[:-1]).T
    L_22 = K_22 - L_21 @ L_21.T
    L_22[L_22 < 0] = 0
    L_22 = np.sqrt(L_22)
    prob.param_dict['L_21_22'].value = np.hstack([L_21, L_22])
    
    prob.param_dict['lambda'].value = np.asarray([[1/radius]])
    
    return prob

def compute_adaptive_threshold(
    prob,
    scores_calib : np.ndarray, 
    x_calib : np.ndarray,
    x_test : np.ndarray, 
    kernel : str, 
    gamma : float = 1,
    M : float = None,
    radius : float = 1
):
    prob = finish_dual_setup(
                prob,
                scores_calib, 
                x_calib,
                x_test, 
                kernel, 
                gamma,
                M,
                radius
            )
    prob.solve(
        solver='MOSEK', 
        verbose=False, 
        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
    )

    var_dict = {}
    var_dict['weights'] = prob.var_dict['weights'].value
    if 'intercept' in prob.var_dict:
        var_dict['intercept'] = prob.var_dict['intercept'].value
    else:
        var_dict['intercept'] = prob.constraints[2].dual_value

    return var_dict

def compute_shifted_coverage(
    scores_test : np.ndarray, 
    scores_calib : np.ndarray, 
    x_calib : np.ndarray, 
    x_test : np.ndarray,
    shift_loc : float, 
    kernel : str, 
    alpha : float, 
    radius : float,
    gamma : float
):        
    obs_coverages = []
    thresholds = []
    
    
    ### Get estimated Coverage
    prob_calib = setup_cvx_dual(
        x_calib[:-1,:],
        scores_calib[:-1],
        kernel=kernel,
        gamma=gamma,
        alpha=alpha
    )
    f_hat = compute_adaptive_threshold(
        prob_calib,
        scores_calib[:-1],
        x_calib[:-1,:],
        x_calib[-1,:],
        kernel=kernel,
        gamma=gamma,
        radius=radius
    )

    g = pairwise_kernels(
        X=np.array([shift_loc]).reshape(1,-1),
        Y = x_calib,
        metric=kernel,
        gamma=gamma
    )

    inner_product = g @ f_hat['weights']
    g_sum = np.mean((g))
    est_coverage = alpha - (1 / radius) * (inner_product / g_sum)

    ### Compute empircal coverage
    prob = setup_cvx_dual(
        x_calib,
        scores_calib,
        kernel=kernel,
        gamma=gamma,
        alpha=alpha
    )
    
    for i in tqdm(range(len(x_test))):
        f_hat = compute_adaptive_threshold(
            prob,
            scores_calib,
            x_calib,
            x_test[i,:],
            kernel=kernel,
            gamma=gamma,
            radius=radius
        )

        K_test = pairwise_kernels(
            X=x_test[i,:].reshape(1,-1),
            Y=np.concatenate([x_calib, x_test[i,:].reshape(1,-1)], axis=0),
            metric=kernel,
            gamma=gamma
        )
        score_threshold = f_hat['intercept'] + K_test @ f_hat['weights']
        thresholds.append(score_threshold)
        coverage = int(scores_test[i] < score_threshold)
        obs_coverages.append(coverage)

    g = pairwise_kernels(
        X=np.array([shift_loc]).reshape(1,-1),
        Y=x_test,
        metric=kernel,
        gamma=gamma
    )[0,:]
    return est_coverage[0], np.average(obs_coverages, weights=g), np.asarray(thresholds).flatten()

def runCV(x_calib,scores_calib,kernel,gamma,alpha,k,min_lamb = None,max_lamb=1, num_lamb = 20):
    if min_lamb == None:
        min_lamb = 1/len(scores_calib)
    lambdas = np.linspace(min_lamb,max_lamb,num_lamb)
        
    folds = KFold(n_splits = k)
    cvxProblemList = []
    Klist = []
    for i, (train_index, test_index) in enumerate(folds.split(x_calib)):
        prob = setup_full_cvx_dual(
                x_calib[train_index,:],
                scores_calib[train_index],
                kernel=kernel,
                gamma=gamma,
                alpha=alpha
            )
        cvxProblemList.append(prob)
        Klist.append(pairwise_kernels(
            X=x_calib[test_index,:],
            Y=x_calib[train_index,:],
            metric=kernel,
            gamma=gamma
        ))
        
        
    allLosses = np.zeros(len(lambdas))
    countR = 0
    for lamb in tqdm(lambdas):
        i=0
        for i, (train_index, test_index) in enumerate(folds.split(x_calib)):        
            cvxProblemList[i].param_dict['lambda'].value = np.asarray([[lamb]])
            cvxProblemList[i].solve(
                solver='MOSEK', 
                verbose=False, 
                mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
            )
            intercept = cvxProblemList[i].constraints[2].dual_value
            resid = (scores_calib[test_index] -
                    Klist[i] @ cvxProblemList[i].var_dict['weights'].value -
                    intercept)
            loss = sum(0.5 * np.abs(resid) + (alpha - 0.5)*resid) * (1/(len(test_index)))
    
            allLosses[countR] = allLosses[countR] + loss/k
            
        countR = countR + 1
        
        
    return allLosses, lambdas
            

        
