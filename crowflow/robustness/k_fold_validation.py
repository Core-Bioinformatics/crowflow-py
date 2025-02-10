from ..clustering.repeated_stochastic_clustering import StochasticClusteringRunner
from sklearn.model_selection import KFold
import ClustAssessPy as ca


class KFoldClusteringValidator:
    """
    Performs k-fold validation to assess clustering stability.

    This class evaluates how stable clustering assignments remain across different 
    data partitions by comparing clustering results on k-fold subsets with those 
    from the full dataset. Element-Centric Similarity (ECS) is used to quantify 
    consistency between fold-level clustering and the baseline (full dataset).

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g., KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the random seed.
    k_folds : int, optional (default=5)
        Number of k-folds for cross-validation.
    n_runs : int, optional (default=30)
        Number of repeated runs per clustering iteration.
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
    >>> kfold_validator = KFoldClusteringValidator(
    >>>     clustering_algo=KMeans,
    >>>     parameter_name_seed="random_state",
    >>>     k_folds=5,
    >>>     n_runs=30,
    >>>     n_clusters=3,
    >>>     init="random"
    >>> )
    >>> baseline_results, kfolds_robustness_results = kfold_validator.run(df)
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        k_folds=5,
        n_runs=30,
        verbose=False,
        labels_name=None,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.k_folds = k_folds
        self.n_runs = n_runs
        self.verbose = verbose
        self.labels_name = labels_name
        self.kwargs = kwargs

    def run(self, data):
        """
        Executes k-fold clustering validation and computes ECS scores.

        The function first clusters the full dataset to establish a baseline solution. 
        It then splits the dataset into k folds and runs clustering on each fold’s training set. 
        The clustering results from each fold are compared to the corresponding subset 
        of the full dataset using Element-Centric Similarity (ECS).

        Parameters
        ----------
        data : array-like
            Dataset to evaluate clustering stability.

        Returns
        -------
        dict
            - 'baseline_results': Clustering results from the full dataset.
            - 'kfold_results': Dictionary containing per-fold clustering validation:
                - 'stochastic_clustering_results': Majority-voting labels for the fold.
                - 'used_indices': Indices of training data for the fold.
                - 'leave_out_indices': Indices excluded from the fold.
                - 'el_score_vector': ECS scores comparing fold labels with full dataset clustering.
        """
        runner = StochasticClusteringRunner(
            self.clustering_algo,
            self.parameter_name_seed,
            n_runs=self.n_runs,
            verbose=self.verbose,
            labels_name=self.labels_name,
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

            kfolds_robustness_results[f"fold_{i + 1}"] = {
                "stochastic_clustering_results": result,
                "used_indices": used_index,
                "leave_out_indices": leave_out_index,
                "el_score_vector": el_score_vector,
            }

        return baseline_results, kfolds_robustness_results
