import cvxpy as cp
import numpy as np

from functools import partial
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Callable

FUNCTION_DEFAULTS = {"kernel": None, "gamma" : 1, "lambda": 1}

class CondConf:
    def __init__(
            self, 
            score_fn : Callable,
            Phi_fn : Callable,
            infinite_params : dict = {}
        ):
        """
        Constructs the CondConf object that caches relevant information for
        generating conditionally valid prediction sets.

        We define the score function and set of conditional guarantees
        that we care about in this function.

        Parameters
        ---------
        score_fn : Callable[np.ndarray, np.ndarray] -> np.ndarray
            Fixed (vectorized) conformity score function that takes in
            X and Y as inputs and returns S as output

        Phi_fn : Callable[np.ndarray] -> np.ndarray
            Function that defines finite basis set that we provide
            exact conditional guarantees over
        
        infinite_params : dict = {}
            Dictionary containing parameters for the RKHS component of the fit
            Valid keys are ('kernel', 'gamma', 'lambda')
                'kernel' should be a valid kernel name for sklearn.metrics.pairwise_kernels
                'gamma' is a hyperparameter for certain kernels
                'lambda' is the regularization penalty applied to the RKHS component
        """
        self.score_fn = score_fn
        self.Phi_fn = Phi_fn
        self.infinite_params = infinite_params

    def setup_problem(
            self,
            alpha : float,
            x_calib : np.ndarray,
            y_calib : np.ndarray
    ):
        """
        setup_problem sets up the final fitting problem for a given quantile alpha
        and a calibration set

        The resulting cvxpy Problem object is stored inside the CondConf parent.

        Arguments
        ---------
        alpha : float 
            Nominal quantile we are fitting

        x_calib : np.ndarray
            Covariate data for the calibration set

        y_calib : np.ndarray
            Labels for the calibration set

        """
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.phi_calib = self.Phi_fn(x_calib)
        self.scores_calib = self.score_fn(x_calib, y_calib)

        self.cvx_problem = setup_cvx_problem(
            alpha,
            self.x_calib,
            self.scores_calib,
            self.phi_calib,
            self.infinite_params
        )

        self.alpha = alpha
        

    def predict(
            self,
            x_test : np.ndarray,
            score_inv_fn : Callable,
            S_min : float = None,
            S_max : float = None
    ):
        """
        Returns the (conditionally valid) prediction set for a given 
        test point

        Arguments
        ---------
        x_test : np.ndarray
            Single test point
        score_inv_fn : Callable[float, np.ndarray] -> .
            Function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*
        S_min : float = None
            Lower bound (if available) on the conformity scores
        S_max : float = None
            Upper bound (if available) on the conformity scores

        Returns
        -------
        prediction_set
        """
        scores_calib = self.score_fn(self.x_calib, self.y_calib)
        if S_min is None:
            S_min = np.min(scores_calib)
        if S_max is None:
            S_max = np.max(scores_calib)

        _solve = partial(_solve_dual, gcc=self, x_test=x_test)

        sol = find_first_zero(_solve, S_min, S_max * 2)

        threshold = self._get_threshold(sol, x_test)

        return score_inv_fn(threshold, x_test.reshape(-1,1))

    def estimate_coverage(
            self,
            nominal_alpha : float,
            weights : np.ndarray,
            x : np.ndarray = None
    ):
        """
        estimate_coverage estimates the true percentile of the issued estimate of the
        conditional quantile under the covariate shift induced by 'weights'

        If we are ostensibly estimating the 0.95-quantile using an RKHS fit, we may 
        determine using our theory that the true percentile of this estimate is only 0.93

        Arguments
        ---------
        nominal_alpha : float
            Nominal quantile level
        weights : np.ndarray
            RKHS weights for tilt under which the coverage is estimated
        x : np.ndarray = None
            Points for which the RKHS weights are defined. If None, we assume
            that weights corresponds to x_calib

        Returns
        -------
        estimated_alpha : float
            Our estimate for the realized quantile level
        """
        weights = weights.reshape(-1,1)
        prob = setup_cvx_problem_calib(
            nominal_alpha,
            self.x_calib,
            self.scores_calib,
            self.phi_calib,
            self.infinite_params
        )
        prob.solve(solver="MOSEK")

        fitted_weights = prob.var_dict['weights'].value
        if x is not None:
            K = pairwise_kernels(
                X=x,
                Y=self.x_calib,
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
        else:
            K = pairwise_kernels(
                X=self.x_calib,
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
        inner_prod = weights.T @ K @ fitted_weights
        expectation = np.mean(weights.T @ K)
        penalty = self.infinite_params['lambda'] * (inner_prod / expectation)
        return nominal_alpha - penalty
    
    def predict_naive(
            self,
            alpha : float,
            x : np.ndarray,
            score_inv_fn : Callable
    ):
        """
        If we do not wish to include the imputed data point, we can sanity check that
        the regression is appropriately adaptive to the conditional variability in the data
        by running a quantile regression on the calibration set without any imputation. 
        When n_calib is large and the fit is stable, we expect these two sets to nearly coincide.

        Arguments
        ---------
        alpha : float
            Nominal quantile we are estimating
        x : np.ndarray
            Set of points for which we are issuing prediction sets
        score_inv_fn : Callable[np.ndarray, np.ndarray] -> np.ndarray
            Vectorized function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*
        
        Returns
        -------
        prediction_sets
        
        """
        if len(x.shape) < 2:
            raise ValueError("x needs to have shape (m, n), not {x_test.shape}.")

        prob = setup_cvx_problem_calib(
            alpha,
            self.x_calib,
            self.scores_calib,
            self.phi_calib,
            self.infinite_params
        )
        prob.solve(solver="MOSEK")

        var_dict = {}
        var_dict['weights'] = prob.var_dict['weights'].value
        var_dict['c0'] = prob.constraints[-1].dual_value
        threshold = self.Phi_fn(x) @ var_dict['c0']
        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
            K = pairwise_kernels(
                X=x,
                Y=self.x_calib,
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
            threshold = (K @ var_dict['weights']) + threshold

        return score_inv_fn(threshold, x)
    
    def verify_coverage(
            self,
            x : np.ndarray,
            y : np.ndarray
    ):
        """
        In some experiments, we may simply be interested in verifying the coverage of our method.
        In this case, we do not need to binary search for the threshold S^*, but only need to verify that
        S <= f_S(x) for the true value of S. This function implements this check for test points
        denoted by x and y

        Arguments
        ---------
        x : np.ndarray
            A vector of test covariates
        y : np.ndarray
            A vector of test labels

        Returns
        -------
        coverage_booleans : np.ndarray
        """
        covers = []
        for x_val, y_val in zip(x, y):
            S_true = self.score_fn(x_val.reshape(-1,1), y_val)
            threshold = self._get_threshold(S_true[0], x_val.reshape(-1,1))
            covers.append(S_true <= threshold)

        return np.asarray(covers)
    
    def _get_threshold(
        self,
        S : float,
        x : np.ndarray
    ):
        prob = finish_dual_setup(
            S,
            self.cvx_problem,
            x,
            self.Phi_fn(x),
            self.x_calib,
            self.infinite_params
        )
        prob.solve(solver="MOSEK")

        var_dict = {}
        var_dict['weights'] = prob.var_dict['weights'].value
        var_dict['c0'] = prob.constraints[-1].dual_value
        threshold = self.Phi_fn(x) @ var_dict['c0']
        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
            K = pairwise_kernels(
                X=np.concatenate([self.x_calib, x.reshape(1,-1)], axis=0),
                Y=np.concatenate([self.x_calib, x.reshape(1,-1)], axis=0),
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
            threshold = (K @ var_dict['weights'])[-1] + threshold
        return threshold


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

def setup_cvx_problem_calib(
    alpha,
    x_calib, 
    scores_calib, 
    phi_calib,
    infinite_params = {}
):
    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib,1))
        
    eta = cp.Variable(name="weights", shape=n_calib)
        
    scores = cp.Constant(scores_calib.reshape(-1,1))
    
    Phi = cp.Constant(phi_calib)

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
        radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])

        _, L = _get_kernel_matrix(x_calib, kernel, gamma)
    
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