# condconformal

`condconformal` is a Python package for conformal prediction with 
conditional guarantees.

For example, given a collection of groups $\mathcal{G}$, `condconformal` issues
a prediction set $\hat{C}(\cdot)$ satisfying

$$\mathbb{P}(Y_{n + 1} \in \hat{C}(X_{n + 1}) \mid X \in G) \geq 1 - \alpha.$$ 

## Installation

condconformal can be installed (locally for now) with pip.

To install with pip, navigate to the directory for this repo and

```bash
$ pip install . 
```

## Examples

The easiest way to start using fairaudit may be to go through the following notebook:

 * [Synthetic Data](https://github.com/jjcherian/conditional-conformal/blob/main/experiments/SyntheticData.ipynb)


## Usage

### The `gcc` Class

The `gcc` class has the following API:
```python
class gcc:
    def __init__(
	self, 
	x: np.ndarray, 
	y: np.ndarray, 
	z: np.ndarray, 
	metric: fairaudit.Metric): ...

    def calibrate_groups(
        self, 
        alpha : float,
        type : str,
        groups : Union[np.ndarray, str],
        epsilon : float = None,
        bootstrap_params : dict = {}
    ) -> None:
        """
        Obtain bootstrap critical values for a specific group collection.

        Parameters
        ----------
        alpha : float
            Type I error threshold
        type : str
            Takes one of three values ('lower', 'upper', 'interval').
            See epsilon documentation for what these correpsond to.
        groups : Union[np.ndarray, str]
            Either a string for a supported collection of groups or a numpy array
            likely obtained by calling `get_intersections` or `get_rectangles` 
            from group.py
            Array dimensions should be (n_points, n_groups)
        epsilon : float = None
            epsilon = None calibrates for issuing confidence bounds. 
                type = "upper" issues lower confidence bounds, 
                type = "lower" issues upper confidence bounds, 
                type = "interval" issues confidence intervals.
            If a non-null value is passed in, we issue a Boolean certificate. 
                type = "upper" tests the null that epsilon(G) >= epsilon
                type = "lower" tests the null that epsilon(G) <= epsilon
                type = "interval" tests the null that |epsilon(G)| >= epsilon
        bootstrap_params : dict = {}
            Allows the user to specify a random seed, number of bootstrap resamples,
            and studentization parameters for the bootstrap process.
        """

    def query_group(self, group : Union[np.ndarray, int]) -> 
    Tuple[List[Union[float, bool]], List[float], List[float]]:
        """
        Query calibrated auditor for certificate for a particular group
        
        Parameters
        ----------
        group : Union[np.ndarray, int]
            Will accept index into groups originally passed in or Boolean 
            array if calibrated collection was infinite

        Returns
        -------
        certificate : List[Union[float, bool]]
            Boolean certificates or confidence bounds for each metric audited
        value : List[float]
            Empirical value of epsilon(G) for each metric audited
        threshold : List[float]
            Estimate of theta for each metric audited
        """

    def calibrate_rkhs(
        self,
        alpha : float,
        type : str,
        kernel : str,
        kernel_params : dict = {},
        bootstrap_params : dict = {}
    ) -> None:
        """
        Obtain bootstrap critical value for a specified RKHS.

        Parameters
        ----------
        alpha : float
            Type I error threshold
        type : str
            Takes one of three values ('lower', 'upper', 'interval').
                type = "upper" issues lower confidence bounds, 
                type = "lower" issues upper confidence bounds, 
                type = "interval" issues confidence intervals.
        kernel : str
            Name of scikit-learn kernel the user would like to use. 
            Suggested kernels: 'rbf' 'laplacian' 'sigmoid'
        kenrnel_params : dict = {}
            Additional parameters required to specify the kernel, 
            e.g. {'gamma': 1} for RBF kernel
        bootstrap_params : dict = {}
            Allows the user to specify a random seed, number of bootstrap resamples,
            and studentization parameters for the bootstrap process.
        """

    def query_rkhs(self, weights : np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Query calibrated auditor for certificate for a particular RKHS
        function.
        
        Parameters
        ----------
        weights : np.ndarray
            RHKS function weights, i.e. f(x_i) = (Kw)_i
        Returns
        -------
        certificate : List[float]
            Confidence bounds for each metric queried.
        value : List[float]
            Empirical value of epsilon(G) for each metric queried.
        """

    def flag_groups(
        self,
        groups : np.ndarray,
        type : str,
        alpha : float,
        epsilon : float = 0,
        bootstrap_params : dict = {"student" : "mad", "student_threshold" : -np.inf}  
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns flags and estimates of epsilon(G) for each group in some finite 
        collection. 

        Parameters
        ----------
        groups : np.ndarray
            Boolean numpy array of dimension (n_points, n_groups)
        type : str
            Takes values ('upper', 'lower', 'interval')
            'upper' tests the null, epsilon(G) >= epsilon
            'lower' tests the null, epsilon(G) <= epsilon
            'interval' tests the null, |epsilon(G)| <= epsilon
        alpha : float
            FDR level
        epsilon : float = 0
            See 'type' documentation
        bootstrap_params : dict = {"student" : "mad", "student_threshold" : -np.inf} 
            Allows the user to specify a random seed, number of bootstrap resamples,
            and studentization parameters for the bootstrap process.
        
        Returns
        -------
        flags : List[bool]
            One flag is raised for each group - at least one metric must be flagged 
	    for the group to receive a True flag.
        values : List[float]
            Empirical value of epsilon(G) for each metric queried.
        """
```


### The `Metric` class

The Metric object should be instantiated for any performance metrics the auditor 
may be interested in querying. Its methods are never directly queried by the user, 
but its constructor should be usable.

```python
class Metric:
    def __init__(
        self, 
        name : str, 
        evaluation_function : Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
        threshold_function : Callable[[np.ndarray, np.ndarray], float] = None,
        metric_params : dict = {}
    ) -> None:
        """
        Constructs the Metric object used by the Auditor object to recompute performance
        over bootstrap samples.

        Parameters
        ----------
        name : str
        evaluation_function : Callable[[np.ndarray, np.ndarray], np.ndarray] = None
            Function applied to model predictions (Z) and true labels (Y) that returns 
            an array of metric values, e.g. the evaluation_function for 
            mean squared error is lambda z, y: return (z - y)**2
        threshold_function : Callable[[np.ndarray, np.ndarray], float]
            Function applied to model predictions (Z) and true labels (Y) that returns 
            a single threshold for comparison, e.g. when comparing to the population average, 
            the threshold_function for MSE is lambda z, y: return np.mean((z - y)**2)
        metric_params : dict = {}
            Additional parameters may be required for metrics that require separate error
            tracking depending on the predicted value or true label.

            For 'calibration'-type metrics, the key 'calibration_bins' should map to a list
            that determines how the predicted values (Z) should be digitized/binned

            For 'equalized odds'-type metrics, the key 'y_values' should map to a list
            so that the metric is calculated separately for each value of y in that list
        """
```


### The `groups` module
We provide two methods in the `groups` module for constructing collections of groups
that can be audited.

```python
def get_intersections(
    X : np.ndarray, 
    discretization : dict = {},
    depth : int = None
) -> np.ndarray:
    """
    Construct groups formed by intersections of other attributes.
    
    Parameters
    ----------
    X : np.ndarray
    discretization : dict = {}
        Keys index columns of X
        Values specify input to the "bins" argument of np.digitize(...)
    depth : int = None
        If None, we consider all intersections, otherwise
        we all consider intersections of up to specified depth.
    Returns
    ---------
    groups : np.ndarray
        Boolean numpy array of size (n_points, n_groups)
    """

def get_rectangles(X : np.ndarray, discretization : dict = {}) -> np.ndarray:
    """
    Construct rectangles formed by attributes.

    Parameters
    ----------

    discretization : dict 
        Keys index columns of X
        Values specify input to the "bins" argument of np.digitize(...)

    Returns
    ---------
    groups : np.ndarray
        Boolean numpy array of size (n_points, n_groups)
    """
```


# Citing
If you use this code in a research project, please cite the forthcoming paper. 
```
@article{gibbs2023conformal,
    title={Conformal Prediction with Conditional Guarantees},
    author={Isaac Gibbs, John Cherian, Emmanuel Cand\`es},
    publisher = {arXiv},
    year = {2023},
    note = {arXiv:XXXX.XXXX [stat.ME]},
    url = {https://arxiv.org/abs/XXXX.XXXXX},
}
``` 
