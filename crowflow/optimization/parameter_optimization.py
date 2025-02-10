from ..clustering.repeated_stochastic_clustering import StochasticClusteringRunner
from itertools import product
import numpy as np
import pandas as pd
import inspect


class ParameterOptimizer:
    """
    Optimizes individual hyperparameters of a stochastic clustering algorithm.

    This class systematically tunes each hyperparameter separately by performing
    repeated clustering and evaluating stability using Element-Centric Consistency (ECC).
    It's purpose is to help identify the value of each parameter that maximizes clustering robustness.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g., KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the random seed.
    parameters_optimize_dict : dict
        Dictionary specifying the parameters and their candidate values,
        e.g., {'n_clusters': [2, 3, 4]}.
    n_runs : int, optional (default=30)
        Number of clustering runs per parameter setting.
    verbose : bool, optional (default=False)
        If True, prints progress updates.
    labels_name : str, optional
        Name of the attribute in the clustering result that contains labels.
        If None, we assume the function directly returns labels.
    **kwargs :
        Additional parameters for the clustering algorithm.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
    >>> parameter_optimizer = ParameterOptimizer(
    >>>     clustering_algo=KMeans,
    >>>     parameter_name_seed='random_state',
    >>>     parameters_optimize_dict={'n_clusters': np.arange(2, 5, 1)},
    >>>     n_runs=30
    >>> )
    >>> results_df, scr_results = parameter_optimizer.run(df)
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        parameters_optimize_dict,
        n_runs=30,
        verbose=False,
        labels_name=None,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.parameters_optimize_dict = parameters_optimize_dict
        self.n_runs = n_runs
        self.verbose = verbose
        self.labels_name = labels_name
        self.kwargs = kwargs
        self._validate_clustering_algo()

    def _validate_clustering_algo(self):
        try:
            signature = inspect.signature(self.clustering_algo)
        except TypeError:
            raise ValueError(
                f"The provided clustering_algo must be callable. Got: {type(self.clustering_algo)}"
            )
        if self.parameter_name_seed not in signature.parameters:
            raise ValueError(
                f"The algorithm {self.clustering_algo.__name__} does not accept a '{self.parameter_name_seed}' parameter."
            )

    def run(self, data):
        """
        Runs parameter optimization by evaluating ECC for each candidate value.

        Parameters
        ----------
        data : array-like
            The dataset on which clustering is performed.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row represents a parameter-value combination
            along with ECC scores and the median ECC.
        dict
            A dictionary containing clustering results (`StochasticClusteringRunner`)
            for each parameter setting.
        """
        ecc_results = []
        stochastic_clustering_results = {}

        for param_name, values_to_try in self.parameters_optimize_dict.items():
            for val in values_to_try:
                if self.verbose:
                    print(f"Running with {param_name}={val}")

                runner = StochasticClusteringRunner(
                    self.clustering_algo,
                    self.parameter_name_seed,
                    n_runs=self.n_runs,
                    verbose=False,
                    labels_name=self.labels_name,
                    **{param_name: val, **self.kwargs},
                )

                result = runner.run(data)
                ecc = result["ecc"]
                median_ecc = np.median(ecc)

                if self.verbose:
                    print("\n Median ECC: ", median_ecc)
                    print(
                        "--------------------------------------------------------------"
                    )

                ecc_results.append(
                    {
                        "param": f"{param_name}_{val}",
                        "ecc": ecc,
                        "median_ecc": median_ecc,
                    }
                )
                stochastic_clustering_results[f"{param_name}_{val}"] = result

        return pd.DataFrame(ecc_results), stochastic_clustering_results


class ParameterSearcher:
    """
    Performs a full grid search over multiple clustering hyperparameters.

    This class evaluates all possible combinations of specified parameters,
    running repeated clustering and computing Element-Centric Consistency (ECC)
    for each combination. It's purpose is to help identify the optimal parameter set that maximizes
    clustering stability.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g., KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the random seed.
    param_grid : dict
        Dictionary of parameters and their values, e.g.,
        {'n_clusters': [2, 3, 4], 'init': ['k-means++', 'random']}.
    n_runs : int, optional (default=30)
        Number of clustering runs per parameter combination.
    verbose : bool, optional (default=False)
        If True, prints progress updates.
    labels_name : str, optional
        Name of the attribute in the clustering result that contains labels.
        If None, we assume the function directly returns labels.
    **kwargs :
        Additional parameters for the clustering algorithm.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
    >>> param_grid = {
    >>>     'n_clusters': np.arange(2, 5, 1),
    >>>     'init': ['k-means++', 'random']
    >>> }
    >>> parameter_searcher = ParameterSearcher(
    >>>     clustering_algo=KMeans,
    >>>     parameter_name_seed='random_state',
    >>>     param_grid=param_grid,
    >>>     n_runs=30
    >>> )
    >>> results_df, scr_results = parameter_searcher.run(df)
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        param_grid,
        n_runs=30,
        verbose=False,
        labels_name=None,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.param_grid = param_grid
        self.n_runs = n_runs
        self.verbose = verbose
        self.kwargs = kwargs
        self.param_combinations = list(product(*param_grid.values()))
        self.param_names = list(param_grid.keys())
        self.labels_name = labels_name
        self._validate_clustering_algo()

    def _validate_clustering_algo(self):
        try:
            signature = inspect.signature(self.clustering_algo)
        except TypeError:
            raise ValueError(
                f"The provided clustering_algo must be callable. Got: {type(self.clustering_algo)}"
            )
        if self.parameter_name_seed not in signature.parameters:
            raise ValueError(
                f"The algorithm {self.clustering_algo.__name__} does not accept a '{self.parameter_name_seed}' parameter."
            )

    def run(self, data):
        """
        Runs a parameter grid search by evaluating all possible parameter combinations.

        Parameters
        ----------
        data : array-like
            The dataset on which clustering is performed.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row corresponds to a parameter combination 
            with ECC scores and the median ECC.
        dict
            A dictionary containing clustering results (`StochasticClusteringRunner`) 
            for each parameter combination tested.
        """
        ecc_results = []
        stochastic_clustering_results = {}

        for param_set in self.param_combinations:
            param_dict = dict(zip(self.param_names, param_set))
            if self.verbose:
                print(f"Testing parameters: {param_dict}")

            runner = StochasticClusteringRunner(
                self.clustering_algo,
                self.parameter_name_seed,
                n_runs=self.n_runs,
                verbose=False,
                labels_name=self.labels_name,
                **{**self.kwargs, **param_dict},
            )
            result = runner.run(data)
            ecc = result["ecc"]
            median_ecc = np.median(ecc)

            if self.verbose:
                print("\n Median ECC: ", median_ecc)
                print("--------------------------------------------------------------")

            ecc_results.append(
                {"params": param_dict, "ecc": ecc, "median_ecc": median_ecc}
            )

            param_str = "_".join([f"{k}_{v}" for k, v in param_dict.items()])
            stochastic_clustering_results[param_str] = result

        return pd.DataFrame(ecc_results), stochastic_clustering_results
