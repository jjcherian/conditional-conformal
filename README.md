# Conditional Conformal

`conditionalconformal` is a Python package for conformal prediction with 
conditional guarantees.

For example, given a collection of groups $\mathcal{G}$, `conditionalconformal` issues
a prediction set $\hat{C}(\cdot)$ satisfying

$$\mathbb{P}(Y_{n + 1} \in \hat{C}(X_{n + 1}) \mid X \in G) \geq 1 - \alpha \quad \text{for all $G \in \mathcal{G}$}.$$ 

Alternatively, given a collection of covariate shifts $\mathcal{F}$, the package issues
a prediction set $\hat{C}(\cdot)$ satisfying 

$$\mathbb{P}_ f(Y_{n + 1} \in \hat{C}(X_{n + 1})) \geq 1 - \alpha \quad \text{for all $f \in \mathcal{F}$}.$$ 


If the collection of shifts is unknown, we also provide a methodology
for providing finite-sample guarantees over arbitrary shifts. To query for the guarantee (which can
depend on the shift of interest), we provide the `estimate_coverage` function.

## Installation

conditionalconformal can be installed with pip.

```bash
$ pip install conditionalconformal
```

## Examples

The easiest way to start using conditionalconformal may be to go through the following notebook:

 * [Synthetic Data](https://github.com/jjcherian/conditional-conformal/blob/release/experiments/SyntheticData.ipynb)


## Usage

### The `CondConf` Class

The `CondConf` class has the following API:
```python
class CondConf:
    def __init__(
            self, 
            score_fn : Callable,
            Phi_fn : Callable,
	    quantile_fn : Callable = None,
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

	quantile_fn : Callable[np.ndarray] -> np.ndarray
	    Function that defines data-dependent quantile we estimate
	    that takes the same input as Phi_fn

        infinite_params : dict = {}
            Dictionary containing parameters for the RKHS component of the fit
            Valid keys are ('kernel', 'gamma', 'lambda')
                'kernel' should be a valid kernel name for sklearn.metrics.pairwise_kernels
                'gamma' is a hyperparameter for certain kernels
                'lambda' is the regularization penalty applied to the RKHS component
        """

    def setup_problem(
            self,
            x_calib : np.ndarray,
            y_calib : np.ndarray
    ):
        """
        setup_problem sets up the final fitting problem for a 
        particular calibration set

        The resulting cvxpy Problem object is stored inside the CondConf parent.

        Arguments
        ---------
        x_calib : np.ndarray
            Covariate data for the calibration set

        y_calib : np.ndarray
            Labels for the calibration set
        """

    def predict(
            self,
            quantile : float,
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
        quantile : float
            Nominal quantile level / pass in None if quantile_fn is in use
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
    
    def predict_naive(
            self,
            quantile : float,
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
        quantile : float
            Nominal quantile level
        x : np.ndarray
            Set of points for which we are issuing prediction sets
        score_inv_fn : Callable[np.ndarray, np.ndarray] -> np.ndarray
            Vectorized function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*
        
        Returns
        -------
        prediction_sets
        
        """
    
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
```

# Citing
This code is available for use under the MIT license.
If you use this code in a research project, please cite the forthcoming paper. 
```
@article{gibbs2023conformal,
    title={Conformal Prediction with Conditional Guarantees},
    author={Isaac Gibbs, John J. Cherian, Emmanuel J. Cand\`es},
    publisher = {arXiv},
    year = {2023},
    note = {arXiv:2305.12616 [stat.ME]},
    url = {https://arxiv.org/abs/2305.12616},
}
``` 
