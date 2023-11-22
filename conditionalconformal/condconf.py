import numpy as np

import conditionalconformal.lp as lp
import conditionalconformal.qp as qp
from .utilities import get_kernel_matrix

from functools import lru_cache
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Callable

DEFAULT_KERNEL = 'rbf'

class CondConf:
    def __init__(
            self, 
            score_fn : Callable,
            score_inv_fn : Callable,
            Phi_fn : Callable,
            x_calib : np.ndarray,
            y_calib : np.ndarray,
            lambda_ : float = 0,
            **kwargs
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

        score_inv_fn : Callable[float, np.ndarray] -> .
            Function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*

        Phi_fn : Callable[np.ndarray] -> np.ndarray
            Function that defines finite basis set that we provide
            exact conditional guarantees over

        x_calib : np.ndarray
            Calibration set covariates

        y_calib : np.ndarray
            Calibration set response
        
        lambda_ : float
            Regularization penalty for RKHS fit
        
        **kwargs :
            Dictionary containing parameters for the RKHS component of the fit
            Valid keys for sklearn.metrics.pairwise_kernels are ('kernel', 'gamma')
                'kernel' should be a valid kernel name
                'gamma' is a hyperparameter for certain kernels
        """
        self.score_fn = score_fn
        self.score_inv_fn = score_inv_fn
        self.Phi_fn = Phi_fn

        self.x_calib = x_calib
        self.y_calib = y_calib
        self.phi_calib = self.Phi_fn(x_calib)
        self.scores_calib = self.score_fn(x_calib, y_calib)

        self.lambda_ = lambda_
        self.kernel_kwargs = kwargs
        if self.lambda_ > 0:
            if self.kernel_kwargs.get('kernel', None) is None:
                self.kernel_kwargs['kernel'] = DEFAULT_KERNEL
            self.K, self.L = get_kernel_matrix(self.x_calib, **self.kernel_kwargs)
        
    def predict(
            self,
            quantile : float,
            x_test : np.ndarray,
    ):
        """
        Returns the (conditionally valid) prediction set for a given 
        test point

        Arguments
        ---------
        quantile : float
            Nominal quantile level
        x_test : np.ndarray
            Single test point

        Returns
        -------
        prediction_set
        """
        duals, primals = self._get_solution(quantile)
        phi_test = self.Phi_fn(x_test).reshape(-1,1)
        S_test = phi_test.T @ primals

        if self.lambda_ > 0:
            K = pairwise_kernels(
                X=x_test,
                Y=self.x_calib,
                **self.kernel_kwargs
            )
            S_test += K.T @ duals.reshape(-1,1)
            threshold = qp.compute_threshold(
                quantile,
                primals,
                duals,
                self.phi_calib,
                self.scores_calib,
                phi_test,
                S_test
            )
        else:
            threshold = lp.compute_threshold(
                quantile,
                primals,
                duals,
                self.phi_calib,
                self.scores_calib,
                phi_test,
                S_test
            )
        return self.score_inv_fn(threshold, x_test.reshape(-1,1))

    def estimate_coverage(
            self,
            quantile : float,
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
        quantile : float
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

        fitted_weights, _ = self._get_solution(quantile)

        if x is not None:
            K = pairwise_kernels(
                X=x,
                Y=self.x_calib,
                **self.kernel_kwargs
            )
        else:
            K = pairwise_kernels(
                X=self.x_calib,
                **self.kernel_kwargs
            )
        inner_prod = weights.T @ K @ fitted_weights
        expectation = np.mean(weights.T @ K)
        penalty = self.lambda_ * (inner_prod / expectation)
        return quantile - penalty
    
    @lru_cache()
    def _get_solution(
            self,
            quantile : float
    ):
        if self.lambda_ > 0:
            duals, primals = qp.get_solution(self.lambda_, self.K, self.scores_calib, self.phi_calib, quantile)
        else:
            duals, primals = lp.get_solution(self.scores_calib, self.phi_calib, quantile)
        return duals, primals
    
    
    def verify_coverage(
            self,
            x : np.ndarray,
            y : np.ndarray,
            quantile : float
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
        quantile : float
            Nominal quantile level

        Returns
        -------
        coverage_booleans : np.ndarray
        """

        duals, primals = self._get_solution(quantile)
        covers = []
        for x_val, y_val in zip(x, y):
            phi_test = self.Phi_fn(x_val.reshape(-1,1))
            S_interp = phi_test.T @ primals
            S_true = self.score_fn(x_val.reshape(-1,1), y_val)
            if self.lambda_ > 0:
                K = pairwise_kernels(
                    X=x_val.reshape(-1,1),
                    Y=self.x_calib,
                    **self.kernel_kwargs
                )
                S_interp += K.T @ duals

                if S_true < S_interp:
                    cover = True
                else:
                    threshold = qp.compute_threshold(
                        quantile,
                        primals,
                        duals,
                        self.phi_calib,
                        self.scores_calib,
                        phi_test,
                        S_interp
                    )
                    cover = threshold >= S_interp
            else:
                if S_interp < S_true:
                    cover = True
                else:
                    threshold = lp.compute_threshold(
                        quantile,
                        primals,
                        duals,
                        self.phi_calib,
                        self.scores_calib,
                        phi_test,
                        S_interp
                    )
                    cover = threshold >= S_interp
            covers.append(cover)

        return np.asarray(covers)