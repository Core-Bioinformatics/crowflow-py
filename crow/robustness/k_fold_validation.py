from ..clustering.repeated_stochastic_clustering import StochasticClusteringRunner
from sklearn.model_selection import KFold
import ClustAssessPy as ca


class KFoldClusteringValidator:
    """
    Perform k-fold validation on clustering. Splits the data into k-folds, trains on k-1 on each iteration
    and evaluates on the left out set.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g. KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the seed.
    k_folds : int, optional (default=5)
        Number of k-folds.
    n_runs : int, optional (default=30)
        Number of runs for repeated stochastic clustering each time.
    verbose : bool, optional (default=False)
        Print progress if True.
    **kwargs :
        Additional parameters for the clustering algorithm.
    """
    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        k_folds=5,
        n_runs=30,
        verbose=False,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.k_folds = k_folds
        self.n_runs = n_runs
        self.verbose = verbose
        self.kwargs = kwargs

    def run(self, data):
        """
        Run the k-fold CV.

        Parameters
        ----------
        data : array-like
            Original dataset on which to test perturbation robustness.

        Returns
        -------
        dict
            - 'stochastic_clustering_results': The clustering result on the subset of data (fold).
            - 'used_indices': Indices of the data rows used for this fold’s clustering.
            - 'leave_out_indices': Indices of the data rows left out in this fold.
            - 'el_score_vector': A numeric vector of element-centric similarity (ECS) scores comparing 
            the fold’s majority-voting labels to the baseline’s labels on the same indices, 
            as computed by element_sim_elscore from ClustAssessPy.

        Example
        -------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.cluster import KMeans
        >>> np.random.seed(42)
        >>> df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
        >>> kfold_validator = KFoldClusteringValidator(
        >>> clustering_algo=KMeans, 
        >>> parameter_name_seed="random_state", 
        >>>     k_folds=10, 
        >>>     n_runs=30, 
        >>>     n_clusters=3, 
        >>>     init="random"
        >>> )
        >>> baseline_results, kfolds_robustness_results = kfold_validator.run(df)
        """
        runner = StochasticClusteringRunner(
            self.clustering_algo,
            self.parameter_name_seed,
            n_runs=self.n_runs,
            verbose=self.verbose,
            **self.kwargs,
        )

        baseline_results = runner.run(data)
        baseline_majority_labels = baseline_results["majority_voting_labels"]

        kf = KFold(n_splits=self.k_folds)

        kfolds_robustness_results = {}

        for i, (used_index, leave_out_index) in enumerate(kf.split(data)):
            if self.verbose:
                print(f"Fold {i}:")

            # # Cluster only on current fold.
            used_data = data.iloc[used_index]
            result = runner.run(used_data)

            # Subset full data labels only on the current intersection.
            labels_intersection = baseline_majority_labels[used_index]

            # Compare results on current fold with the points on the intesection with the full dataset.
            el_score_vector = ca.element_sim_elscore(
                labels_intersection, result["majority_voting_labels"]
            )

            kfolds_robustness_results[f"fold_{i+1}"] = {
                "stochastic_clustering_results": result,
                "used_indices": used_index,
                "leave_out_indices": leave_out_index,
                "el_score_vector": el_score_vector
            }


        return baseline_results, kfolds_robustness_results


