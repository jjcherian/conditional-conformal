import numpy as np
import cvxpy as cp
import mosek

from tqdm import tqdm

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold

# def setup_cvx_primal(
#     x_calib, scores_calib, kernel, gamma, alpha, radius
# ):    
#     n_calib = len(scores_calib)

#     rkhs_weight = cp.Variable(name="weights", shape=n_calib + 1)
#     intercept = cp.Variable(name="intercept")

#     K_11, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)

#     K_11 = cp.Constant(K_11)
#     K_21 = cp.Parameter(name="K_12", shape=(n_calib, 1))
#     K_12 = cp.Parameter(name="K_21", shape=(1, n_calib + 1))
#     K = cp.hstack([K_11, K_21])
#     K = cp.vstack([K, K_12])

#     scores_const = cp.Constant(scores_calib.reshape(-1,1))
#     scores_param = cp.Parameter(name="score_impute", shape=(1,1))
#     scores = cp.vstack([scores_const, scores_param])

#     L_11_const = cp.Constant(
#         np.hstack([L_11, np.zeros((L_11.shape[0], 1))])
#     )
#     L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
#     L = cp.vstack([L_11_const, L_21_22_param])

#     resid = cp.vec(scores) - (K @ rkhs_weight) - intercept
#     loss = cp.sum(0.5 * cp.abs(resid) + cp.multiply(alpha - 0.5, resid)) * (1/(n_calib + 1))
#     loss += (1/(2*radius)) * cp.sum_squares(L.T @ rkhs_weight)
#     prob = cp.Problem(cp.Minimize(loss))
  
#     return prob

def setup_full_cvx_dual(
    x_calib, scores_calib, kernel, gamma, alpha, z_calib = None
):
    n_calib = len(scores_calib)
    if z_calib is None:
        z_calib = np.ones((n_calib,1))
        
    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    rkhs_weight = cp.Variable(name="weights", shape=n_calib )

    if not kernel is None:
        
        radius = cp.Parameter(name="radius", shape=(1,1))

        _, L = _get_kernel_matrix(x_calib, kernel, gamma)

        L_const = cp.Constant(L)

        C = radius / n_calib
        constraints = [
            C * (alpha - 1) <= rkhs_weight,
            C * alpha >= rkhs_weight,
            rkhs_weight.T @ z_calib == 0]
        prob = cp.Problem(
                            cp.Minimize(0.5 * cp.sum_squares(L_const.T @ rkhs_weight) -
                                        cp.sum(cp.multiply(rkhs_weight, cp.vec(scores_const)))),
                            constraints
                        )
    else:
        constraints = [(alpha - 1) <= rkhs_weight,
                       alpha >= rkhs_weight,
                       rkhs_weight.T @ z_calib == 0]
        prob = cp.Problem(
                            cp.Minimize(-cp.sum(cp.multiply(rkhs_weight, cp.vec(scores_const)))),
                            constraints
                        )

    return prob


def setup_cvx_dual(
    x_calib, scores_calib, kernel, gamma, alpha, z_calib = None
):
    n_calib = len(scores_calib)
    if z_calib is None:
        z_calib = np.ones((n_calib,1))
        
    rkhs_weight = cp.Variable(name="weights", shape=n_calib + 1)
        
    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    scores_param = cp.Parameter(name="score_impute", shape=(1,1))
    scores = cp.vstack([scores_const, scores_param])
    
    z_calibration = cp.Constant(z_calib)
    z_test = cp.Parameter(name="z_test", shape=(1, z_calib.shape[1]))
    Z = cp.vstack([z_calibration, z_test])

    if not kernel is None:

        radius = cp.Parameter(name="radius", shape=(1,1))        

        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
    
        L_11_const = cp.Constant(
            np.hstack([L_11, np.zeros((L_11.shape[0], 1))])
            )
        L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
        L = cp.vstack([L_11_const, L_21_22_param])
    
        C = radius / (n_calib+1)

        constraints = [
            C * (alpha - 1) <= rkhs_weight,
            C * alpha >= rkhs_weight,
            rkhs_weight.T @ Z == 0]
        prob = cp.Problem(
                    cp.Minimize(0.5 * cp.sum_squares(L.T @ rkhs_weight) - cp.sum(cp.multiply(rkhs_weight, cp.vec(scores)))),
                    constraints
                )
    else:
        constraints = [(alpha - 1) <= rkhs_weight,
                       alpha >= rkhs_weight,
                       rkhs_weight.T @ Z == 0]
        prob = cp.Problem(
                            cp.Minimize(-cp.sum(cp.multiply(rkhs_weight, cp.vec(scores)))),
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
    radius : float = 1,
    z_test : np.ndarray = None
):
    if M is None:
        M = np.max(scores_calib)
    prob.param_dict['score_impute'].value = np.asarray([[M]])
    
    prob.param_dict['z_test'].value = z_test

    if not kernel is None:
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
            X=x_test.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )
        L_21 = np.linalg.solve(L_11, K_12[:-1]).T
        L_22 = K_22 - L_21 @ L_21.T
        L_22[L_22 < 0] = 0
        L_22 = np.sqrt(L_22)    
        prob.param_dict['L_21_22'].value = np.hstack([L_21, L_22])
    
        prob.param_dict['radius'].value = np.asarray([[radius]])
    
    return prob

def compute_adaptive_threshold(
    prob,
    scores_calib : np.ndarray, 
    x_calib : np.ndarray,
    x_test : np.ndarray, 
    kernel : str, 
    gamma : float = 1,
    M : float = None,
    radius : float = 1,
    z_calib : np.ndarray = None,
    z_test : np.ndarray = None
):
    if z_calib is None:
        z_calib = np.ones(shape = (len(x_calib),1))
    if z_test is None:
        z_test = np.ones(shape = (len(x_test),1))

    prob = finish_dual_setup(
                prob,
                scores_calib, 
                x_calib,
                x_test, 
                kernel, 
                gamma,
                M,
                radius,
                z_test.reshape((1,z_calib.shape[1]))
            )
    prob.solve(
        solver='MOSEK', 
        verbose=False, 
        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
    )

    var_dict = {}
    var_dict['weights'] = prob.var_dict['weights'].value
    #if 'intercept' in prob.var_dict:
    #    var_dict['intercept'] = prob.var_dict['intercept'].value
    #else:
    var_dict['c0'] = prob.constraints[-1].dual_value
    
    # K = pairwise_kernels(
    #     X=np.concatenate([x_calib, x_test.reshape(1,-1)], axis=0),
    #     Y = np.concatenate([x_calib, x_test.reshape(1,-1)], axis=0),
    #     metric=kernel,
    #     gamma=gamma
    # )
    # z = np.concatenate([z_calib, z_test.reshape(1,-1)], axis=0)
    # thresholds = z@var_dict['c0'] + K@var_dict['weights']
    # s = np.concatenate([scores_calib,np.array([M])])
    
    # print("-----------------------------------------------")
    # print(np.mean(s < thresholds+0.0001) )
    # #print(s-thresholds)
    # print( np.quantile(s-thresholds,0.9))
    # var_dict['c0'][-1] = var_dict['c0'][-1] + np.quantile(s-thresholds,0.9)
    
    # thresholds = z@var_dict['c0'] + K@var_dict['weights']
    # print(np.mean(s < thresholds+0.0001) )
    # print(np.mean(np.abs(thresholds - s)<0.0001))

    return var_dict

def compute_estimated_coverage( 
        scores_calib : np.ndarray, 
        x_calib : np.ndarray, 
        shift_loc : float, 
        kernel : str, 
        alpha : float, 
        radius : float,
        gamma : float,
        z_calib : np.ndarray,
):
    prob_calib = setup_full_cvx_dual(
        x_calib,
        scores_calib,
        kernel=kernel,
        gamma=gamma,
        alpha=alpha,
        z_calib=z_calib
    )
    prob_calib.param_dict['radius'].value = np.asarray([[radius]])
    prob_calib.solve(
        solver='MOSEK', 
        verbose=False, 
        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
    )


    g = pairwise_kernels(
        X=np.array([shift_loc]).reshape(1,-1),
        Y = x_calib,
        metric=kernel,
        gamma=gamma
    )

    inner_product = g @ prob_calib.var_dict['weights'].value
    g_sum = np.mean((g))
    
    return alpha - (1 / radius) * (inner_product / g_sum)



def compute_shifted_coverage( ### Finish editing by incorporating z_calib and z_test
    scores_test : np.ndarray, 
    scores_calib : np.ndarray, 
    x_calib : np.ndarray, 
    x_test : np.ndarray,
    shift_loc : float, 
    kernel : str, 
    alpha : float, 
    radius : float,
    gamma : float,
    z_calib : np.ndarray = None,
    z_test : np.ndarray = None,
    exact = False,
    eps = 0
):        
    if z_calib is None:
        z_calib = np.ones(shape = (len(scores_calib),1))
    if z_test is None:
        z_test = np.ones(shape = (len(scores_test),1))

    
    obs_coverages = []
    thresholds = []
    
    
    ### Get estimated Coverage
    if shift_loc is None:
        est_coverage = None
        g = np.ones(len(scores_test))
    else:
        est_coverage = compute_estimated_coverage(scores_calib, x_calib, shift_loc, 
                                              kernel, alpha, radius, gamma, z_calib)[0]
        g = pairwise_kernels(
                X=np.array([shift_loc]).reshape(1,-1),
                Y=x_test,
                metric=kernel,
                gamma=gamma
            )[0,:]

    ### Compute empircal coverage
    prob = setup_cvx_dual(
        x_calib,
        scores_calib,
        kernel=kernel,
        gamma=gamma,
        alpha=alpha,
        z_calib=z_calib
    )
    
    for i in tqdm(range(len(scores_test))):
        if exact:
            M = scores_test[i]
        else:
            M = None
        f_hat = compute_adaptive_threshold(
            prob,
            scores_calib,
            x_calib,
            x_test[i,:],
            kernel=kernel,
            gamma=gamma,
            radius=radius,
            z_calib = z_calib,
            z_test = z_test[i,:],
            M = M
        )
        if not kernel is None:
            K_test = pairwise_kernels(
                        X=x_test[i,:].reshape(1,-1),
                        Y=np.concatenate([x_calib, x_test[i,:].reshape(1,-1)], axis=0),
                        metric=kernel,
                        gamma=gamma
                    )
            score_threshold = z_test[i,:] @ f_hat['c0'] + K_test @ f_hat['weights']
        else:
            score_threshold = z_test[i,:] @ f_hat['c0']
        thresholds.append(score_threshold)
        coverage = int(scores_test[i] < score_threshold + eps)
        obs_coverages.append(coverage)

    
    return est_coverage, np.average(obs_coverages, weights=g), np.asarray(thresholds).flatten()

def runCV(x_calib,scores_calib,kernel,gamma,alpha,k,min_radius = 1,max_radius=None, num_radii = 20, z_calib=None):
    if z_calib is None:
        z_calib = np.ones(shape = (len(scores_calib),1))
    
    if max_radius == None:
        max_radius = len(scores_calib)
    radii = np.linspace(min_radius,max_radius,num_radii)
        
    folds = KFold(n_splits = k, shuffle = True)
    cvxProblemList = []
    Klist = []
    for i, (train_index, test_index) in enumerate(folds.split(x_calib)):
        prob = setup_full_cvx_dual(
                x_calib[train_index,:],
                scores_calib[train_index],
                kernel=kernel,
                gamma=gamma,
                alpha=alpha,
                z_calib = z_calib[train_index,:]
            )
        cvxProblemList.append(prob)
        Klist.append(pairwise_kernels(
            X=x_calib[test_index,:],
            Y=x_calib[train_index,:],
            metric=kernel,
            gamma=gamma
        ))
        
        
    allLosses = np.zeros(len(radii))
    countR = 0
    for radius in tqdm(radii):
        i=0
        for i, (train_index, test_index) in enumerate(folds.split(x_calib)):        
            cvxProblemList[i].param_dict['radius'].value = np.asarray([[radius]])
            cvxProblemList[i].solve(
                solver='MOSEK', 
                verbose=False, 
                mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
            )
            c0 = cvxProblemList[i].constraints[2].dual_value
            resid = (scores_calib[test_index] -
                    Klist[i] @ cvxProblemList[i].var_dict['weights'].value -
                    z_calib[test_index,:]@c0)
            loss = sum(0.5 * np.abs(resid) + (alpha - 0.5)*resid) * (1/(len(test_index)))
    
            allLosses[countR] = allLosses[countR] + loss/k
            
        countR = countR + 1
        
        
    return allLosses, radii

def compute_qr_coverages(x_calib, x_test, scores_calib, scores_test, alpha, gamma, radius, kernel, z_calib, z_test, eps=0):
    prob = setup_full_cvx_dual(
                x_calib,
                scores_calib,
                kernel=kernel,
                gamma=gamma,
                alpha=alpha,
                z_calib=z_calib
            )
    if not kernel is None:
        prob.param_dict['radius'].value = np.asarray([[radius]])
    prob.solve(
                solver='MOSEK', 
                verbose=False, 
                mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
            )

    if not kernel is None:
        K_test = pairwise_kernels(
                                X=x_test, 
                                Y=x_calib,
                                metric=kernel, 
                                gamma=gamma
                              )
        thresholds = z_test @ prob.constraints[2].dual_value + K_test @ prob.var_dict['weights'].value
    else:
        thresholds = z_test @ prob.constraints[-1].dual_value 

    coverages = scores_test <= thresholds + eps
    
    return np.average(coverages), thresholds

