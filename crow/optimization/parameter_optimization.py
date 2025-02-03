from ..clustering.repeated_stochastic_clustering import StochasticClusteringRunner
from itertools import product
import numpy as np
import pandas as pd
import inspect


class ParameterOptimizer:
    """
    Optimize selected parameters of a stochastic clustering algorithm. Optimizes each parameter
    separately.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g. KMeans from scikit-learn).
    parameter_name_seed : str
        Parameter used to set the seed.
    parameters_optimize_dict : dict
        Dict of {parameter_name: [values]} to try.
    n_runs : int, optional (default=30)
        Number of runs for each parameter setting.
    verbose : bool, optional (default=False)
        Print progress if True.
    **kwargs :
        Additional parameters for the clustering algorithm.
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        parameters_optimize_dict,
        n_runs=30,
        verbose=False,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.parameters_optimize_dict = parameters_optimize_dict
        self.n_runs = n_runs
        self.verbose = verbose
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
        Run the parameter optimization process on the given data.

        Parameters
        ----------
        data : array-like
            The dataset on which the optimization is performed.

        Returns
        -------
        pd.DataFrame
            Each row represents the parameter and value combination along with the resulting element-centric consistency (ECC) scores and median ECC.
        dict
            Dictionary containing the (StochasticClusteringRunner) results for each parameter and value tested.

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
    Parameter grid search using repeated stochastic clustering. Find optimal combinations of
    parameter values.

    Parameters
    ----------
    clustering_algo : callable
        Clustering algorithm class or callable (e.g., KMeans).
    parameter_name_seed : str
        Parameter for the random seed.
    param_grid : dict
        Dict {param_name: [values]} to try all combinations.
    n_runs : int, optional (default=30)
        Number of stochastic runs for each combination.
    verbose : bool, optional (default=False)
        Print progress if True.
    **kwargs :
        Additional arguments for the clustering algorithm.
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        param_grid,
        n_runs=30,
        verbose=False,
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
        Run the parameter search on the given data.

        Parameters
        ----------
        data : array-like
            The dataset on which to run parameter grid search.

        Returns
        -------
        pd.DataFrame
            Each row represents a combination of parameters and the resulting ECC scores and median ECC.
        dict
            Dictionary containing the (StochasticClusteringRunner) results for each combination of parameters tested.
       
        
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
