from ..clustering.repeated_stochastic_clustering import StochasticClusteringRunner
import ClustAssessPy as ca
import numpy as np


class PerturbationRobustnessTester:
    """
    Evaluates the robustness of clustering solutions under feature perturbations.

    This class tests how stable clustering results are when features are altered/perturbed.
    The user must provide a perturbation function, which modifies the dataset before
    clustering is re-run. Stability is assessed using Element-Centric Similarity (ECS)
    between the baseline clustering and perturbation-induced clusterings.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g., KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the random seed.
    perturbation_func : callable
        Function that modifies input data to introduce perturbations.
    n_perturbations : int, optional (default=10)
        Number of perturbation trials.
    n_runs : int, optional (default=30)
        Number of clustering runs per perturbation.
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
    >>> def shuffle_features(X):
    >>>     X_shuffled = X.copy()
    >>>     for col in X_shuffled.columns:
    >>>         X_shuffled[col] = np.random.permutation(X_shuffled[col])
    >>>     return X_shuffled
    >>> df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
    >>> perturb_tester = PerturbationRobustnessTester(
    >>>     clustering_algo=KMeans,
    >>>     parameter_name_seed='random_state',
    >>>     perturbation_func=shuffle_features,
    >>>     n_perturbations=5,
    >>>     n_runs=30
    >>> )
    >>> perturb_results = perturb_tester.run(df)
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        perturbation_func,
        n_perturbations=10,
        n_runs=30,
        verbose=False,
        labels_name=None,
        **kwargs,
    ):
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.perturbation_func = perturbation_func
        self.n_perturbations = n_perturbations
        self.n_runs = n_runs
        self.verbose = verbose
        self.labels_name = labels_name
        self.kwargs = kwargs

    def run(self, data):
        """
        Runs the perturbation robustness test on the dataset.

        The baseline clustering is first computed. Then, the dataset undergoes
        `n_perturbations` perturbations, and clustering is repeated on each
        perturbed dataset. Stability is assessed using Element-Centric Similarity (ECS)
        between the baseline and perturbed cluster assignments.

        Parameters
        ----------
        data : array-like
            The dataset on which clustering is performed.

        Returns
        -------
        dict
            - 'baseline_majority_labels': Majority-vote labels from the baseline clustering.
            - 'baseline_ecc': Element-Centric Consistency (ECC) score of the baseline clustering.
            - 'perturbation_el_sim_scores': List of ECS scores between baseline and perturbed clusterings.
            - 'mean_score': Mean ECS score across all perturbations, indicating overall robustness.
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
        baseline_ecc = baseline_results["ecc"]

        perturbation_el_sim_scores = []
        for i in range(self.n_perturbations):
            if self.verbose:
                print(f"Perturbation {i + 1}/{self.n_perturbations}")
            perturbed_data = self.perturbation_func(data)

            # Run clustering again on perturbed data
            perturbed_results = runner.run(perturbed_data)
            perturbed_majority_labels = perturbed_results["majority_voting_labels"]

            # Calculate element-based similarity between baseline and perturbed majority labels
            score = ca.element_sim(baseline_majority_labels, perturbed_majority_labels)
            perturbation_el_sim_scores.append(score)

        mean_score = np.mean(perturbation_el_sim_scores)

        return {
            "baseline_majority_labels": baseline_majority_labels,
            "baseline_ecc": baseline_ecc,
            "perturbation_el_sim_scores": perturbation_el_sim_scores,
            "mean_score": mean_score,
        }
