import cvxpy as cp
import numpy as np

from functools import partial
from scipy.optimize import root_scalar
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Callable

FUNCTION_DEFAULTS = {"kernel": None, "gamma" : 1, "lambda": 1}

class GCC:
    def __init__(
            self, 
            score_fn : Callable,
            Phi_fn : Callable
        ):
        self.score_fn = score_fn
        self.Phi_fn = Phi_fn

    def set_function_class(
            self,
            alpha : float,
            x_calib : np.ndarray,
            y_calib : np.ndarray,
            infinite_params = {}
    ):
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.phi_calib = self.Phi_fn(x_calib)
        self.scores_calib = self.score_fn(x_calib, y_calib)

        self.cvx_problem = setup_cvx_problem(
            alpha,
            self.x_calib,
            self.scores_calib,
            self.phi_calib,
            infinite_params
        )

        self.alpha = alpha
        self.infinite_params = infinite_params
        

    def predict(
            self,
            x_test : np.ndarray,
            score_inv_fn : Callable,
            S_min : float = None,
            S_max : float = None
    ):
        scores_calib = self.score_fn(self.x_calib, self.y_calib)
        if S_min is None:
            S_min = np.min(scores_calib)
        if S_max is None:
            S_max = np.max(scores_calib)

        _solve = partial(_solve_dual, gcc=self, x_test=x_test)

        sol = find_first_zero(_solve, S_min, S_max * 2)

        prob = finish_dual_setup(
            sol,
            self.cvx_problem,
            x_test,
            self.Phi_fn(x_test),
            self.x_calib,
            self.infinite_params
        )

        prob.solve(solver="MOSEK")

        var_dict = {}
        var_dict['weights'] = prob.var_dict['weights'].value
        var_dict['c0'] = prob.constraints[-1].dual_value
        threshold = self.Phi_fn(x_test) @ var_dict['c0']
        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
            K = pairwise_kernels(
                X=np.concatenate([self.x_calib, x_test.reshape(1,-1)], axis=0),
                Y=np.concatenate([self.x_calib, x_test.reshape(1,-1)], axis=0),
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
            threshold += K @ var_dict['weights']
        return score_inv_fn(threshold, x_test.reshape(-1,1))
    

def find_first_zero(func, min, max, tol=1e-3):
    min, max = float(min), float(max)
    assert (max + tol) > max
    while (max - min) > tol:
        mid = (min + max) / 2
        if np.isclose(func(mid), 0):
            max = mid
        else:
            min = mid
    return max

def _solve_dual(S, gcc, x_test):
    prob = finish_dual_setup(
        S,
        gcc.cvx_problem,
        x_test,
        gcc.Phi_fn(x_test),
        gcc.x_calib,
        gcc.infinite_params
    )
    prob.solve(solver="MOSEK")
    return prob.var_dict['weights'].value[-1] - gcc.alpha
        

def setup_cvx_problem(
    alpha,
    x_calib, 
    scores_calib, 
    phi_calib,
    infinite_params = {}
):
    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib,1))
        
    eta = cp.Variable(name="weights", shape=n_calib + 1)
        
    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    scores_param = cp.Parameter(name="S_test", shape=(1,1))
    scores = cp.vstack([scores_const, scores_param])
    
    Phi_calibration = cp.Constant(phi_calib)
    Phi_test = cp.Parameter(name="Phi_test", shape=(1, phi_calib.shape[1]))
    Phi = cp.vstack([Phi_calibration, Phi_test])

    kernel = infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"])
    gamma = infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])

    if kernel is None: # no RKHS fitting
        constraints = [
            (alpha - 1) <= eta,
            alpha >= eta,
            eta.T @ Phi == 0
        ]
        prob = cp.Problem(
            cp.Minimize(-1 * cp.sum(cp.multiply(eta, cp.vec(scores)))),
            constraints
        )
    else: # RKHS fitting
        radius = cp.Parameter(name="radius")        

        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
    
        L_11_const = cp.Constant(
            np.hstack([L_11, np.zeros((L_11.shape[0], 1))])
            )
        L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
        L = cp.vstack([L_11_const, L_21_22_param])
    
        C = radius / (n_calib + 1)

        constraints = [
            C * (alpha - 1) <= eta,
            C * alpha >= eta,
            eta.T @ Phi == 0]
        prob = cp.Problem(
                    cp.Minimize(0.5 * cp.sum_squares(L.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(scores)))),
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
    S : np.ndarray, 
    prob,
    X : np.ndarray,
    Phi : np.ndarray,
    x_calib : np.ndarray,
    infinite_params = {}
):
    prob.param_dict['S_test'].value = np.asarray([[S]])
    prob.param_dict['Phi_test'].value = Phi.reshape(1,-1)

    kernel = infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
    gamma = infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma'])
    radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])

    if kernel is not None:
        K_12 = pairwise_kernels(
            X=np.concatenate([x_calib, X.reshape(1,-1)], axis=0),
            Y=X.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )

        if 'K_12' in prob.param_dict:
            prob.param_dict['K_12'].value = K_12[:-1]
            prob.param_dict['K_21'].value = K_12.T

        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
        K_22 = pairwise_kernels(
            X=X.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )
        L_21 = np.linalg.solve(L_11, K_12[:-1]).T
        L_22 = K_22 - L_21 @ L_21.T
        L_22[L_22 < 0] = 0
        L_22 = np.sqrt(L_22)    
        prob.param_dict['L_21_22'].value = np.hstack([L_21, L_22])
    
        prob.param_dict['radius'].value = radius    
    
    return prob