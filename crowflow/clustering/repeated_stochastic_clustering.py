from ..utils._private_helper_functions import (
    _get_clustering_labels,
    _reconcile_partitions_and_majority_voting,
)
import numpy as np
import ClustAssessPy as ca
import inspect
from collections import Counter


class StochasticClusteringRunner:
    """
    Repeats stochastic clustering multiple times to assess stability.

    This class runs a clustering algorithm multiple times with different random seeds
    and evaluates the stability of results using Element-Centric Consistency (ECC).
    It identifies in an element-wise precision the stability of clustering results and
    provides a consensus labeling through majority voting.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g., KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the random seed.
    n_runs : int, optional (default=30)
        Number of times the clustering is repeated.
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
    >>> runner = StochasticClusteringRunner(KMeans, "random_state", n_runs=30, verbose=True, n_clusters=3)
    >>> results = runner.run(df)
    >>> print("Majority Voting Labels:", results["majority_voting_labels"])
    >>> print("ECC:", results["ecc"])
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        n_runs=30,
        verbose=False,
        labels_name=None,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
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
                f"The algorithm does not accept a '{self.parameter_name_seed}' parameter. "
                f"Ensure it is a stochastic algorithm (can set random seed)."
            )

    def run(self, data):
        """
        Runs repeated stochastic clustering and evaluates stability.

        This method performs clustering multiple times with different random seeds,
        computes element-wise stability using Element-Centric Consistency (ECC),
        and determines a consensus labeling for each element through majority voting (after
        reconciling the labels using majority voting).

        Parameters
        ----------
        data : array-like
            The dataset on which clustering is performed.

        Returns
        -------
        dict
            - 'partitions': List of partitions (clustering assignments) from each run.
            - 'partition_frequencies': Dictionary mapping partitions to their occurrence frequency.
            - 'majority_voting_labels': Final consensus labels based on majority voting.
            - 'ecc': Element-centric consistency (ECC) scores indicating clustering stability.
            - 'seeds': The list of random seeds used.
        """

        # Generate seeds
        seeds = np.arange(100, 100 + self.n_runs * 100, 100)
        partitions_list = []

        # Run the clustering repeatedly
        for seed in seeds:
            if self.verbose:
                print(f"  Seed={seed}")
            algo_params = {**self.kwargs, self.parameter_name_seed: int(seed)}
            labels = tuple(
                int(label)
                for label in _get_clustering_labels(
                    data, self.clustering_algo, algo_params, self.labels_name
                )
            )
            partitions_list.append(labels)

        # Count identical partitions and their frequency
        partition_counter = Counter(partitions_list)
        partition_frequencies = dict(partition_counter)

        # Get most likely label (majority voting) for each sample
        majority_voting_labels, _ = _reconcile_partitions_and_majority_voting(
            partition_frequencies
        )

        ecc = ca.element_consistency(partitions_list)

        return {
            "partitions": partitions_list,
            "partition_frequencies": partition_frequencies,
            "majority_voting_labels": majority_voting_labels,
            "ecc": ecc,
            "seeds": seeds,
        }
