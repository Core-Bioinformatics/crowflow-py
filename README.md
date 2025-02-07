# crowflow
<p align="center">
  <img src="https://github.com/user-attachments/assets/313177e6-f41d-4ee1-983d-50ce6bd4e719">
</p>

`crowflow` is a Python package designed for assessing clustering stability through repeated stochastic clustering. It is compatible with any clustering algorithm that outputs labels or implements a `fit` or `fit_predict` method, provided it includes stochasticity (i.e., allows setting a seed or `random_state`). By running clustering multiple times with different seeds, `crowflow` quantifies clustering consistency using element-centric similarity (ECS) and element-centric consistency (ECC), offering insights into the robustness and reproducibility of cluster assignments. The package enables users to optimize feature subsets, fine-tune clustering parameters, and evaluate clustering robustness against perturbations.

`crowflow` generalizes the [`ClustAssessPy`](https://github.com/Core-Bioinformatics/ClustAssessPy) package, which focuses on parameter selection for community-detection clustering in single-cell analysis. It extends this approach to any clustering task, enabling a data-driven identification of robust and reproducible clustering solutions across diverse applications.

## Class Summaries

### `StochasticClusteringRunner`
Runs a stochastic clustering algorithm multiple times with different random seeds and evaluates the stability of results using ECC. It identifies in an element-wise precision the stability of clustering results and provides a majority voting label.

#### Example
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from crowflow import StochasticClusteringRunner

np.random.seed(42)
df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
runner = StochasticClusteringRunner(KMeans, "random_state", n_runs=30, verbose=True, n_clusters=3)
results = runner.run(df)
print("Majority Voting Labels:", results["majority_voting_labels"])
print("ECC:", results["ecc"])
```

### `GeneticAlgorithmFeatureSelector`
Uses a genetic algorithm to iteratively optimize feature selection for clustering stability. It repeatedly applies stochastic clustering with different feature subsets and evaluates stability using ECC. The algorithm evolves through selection, crossover, and mutation, converging on the feature set that maximizes clustering robustness.

#### Example
```python
from crowflow import GeneticAlgorithmFeatureSelector

np.random.seed(42)
ga_fs = GeneticAlgorithmFeatureSelector(KMeans, "random_state", verbose=True, n_generations_no_change=5)
ga_results = ga_fs.run(df)
print("Best Features:", ga_results["best_features"])
print("Best ECC Score:", ga_results["best_ecc"])
```

### `ParameterOptimizer`
Systematically tunes each hyperparameter separately by performing repeated clustering and evaluating stability using ECC.

#### Example
```python
from crowflow import ParameterOptimizer

parameter_optimizer = ParameterOptimizer(KMeans, "random_state", {"n_clusters": np.arange(2, 5, 1)}, n_runs=30)
results_df, scr_results = parameter_optimizer.run(df)
```

### `ParameterSearcher`
Evaluates all possible combinations (exhaustive grid search) of specified parameters running repeated clustering and computing ECC for each combination. The purpose is to find the configuration (set of hyperparameter values) that provides the most stable clustering results.

#### Example
```python
from crowflow import ParameterSearcher

param_grid = {"n_clusters": np.arange(2, 5, 1), "init": ["k-means++", "random"]}
parameter_searcher = ParameterSearcher(KMeans, "random_state", param_grid, n_runs=30)
results_df, scr_results = parameter_searcher.run(df)
```

### `KFoldClusteringValidator`
Evaluates how stable clustering assignments remain across different data partitions by comparing clustering results on k-fold subsets with those from the full dataset. ECS is used to quantify similarity/stability between fold-level clustering and the baseline (full dataset).

#### Example
```python
from crowflow import KFoldClusteringValidator

kfold_validator = KFoldClusteringValidator(KMeans, "random_state", k_folds=5, n_runs=30, n_clusters=3, init="random")
baseline_results, kfolds_robustness_results = kfold_validator.run(df)
```

### `PerturbationRobustnessTester`
Tests how stable clustering results are when features are altered/perturbed. The user must provide a perturbation function, which modifies the dataset before clustering is re-run. Stability is assessed using Element-Centric Similarity (ECS) between the baseline clustering and perturbation-induced clusterings.

#### Example
```python
def shuffle_features(X):
    X_shuffled = X.copy()
    for col in X_shuffled.columns:
        X_shuffled[col] = np.random.permutation(X_shuffled[col])
    return X_shuffled

perturb_tester = PerturbationRobustnessTester(KMeans, "random_state", perturbation_func=shuffle_features, n_perturbations=5, n_runs=30)
perturb_results = perturb_tester.run(df)
```

## Installation

`crowflow` requires Python 3.7 or newer.

### Dependencies

- numpy
- matplotlib
- scikit-learn
- seaborn
- plotnine
- ClustAssessPy

### User Installation

We recommend installing `crowflow` in a virtual environment (venv or Conda).

```sh
pip install crowflow
```

## Tutorials

The package can be applied to any clustering task (as long as the clustering algorithm used is stochastic).

In the [cuomo example](examples/cuomo_application.ipynb), we show how to use `crowflow` with `GaussianMixture` from `scikit-learn` to initially assess clustering stability of the default parameter values. We then attempt to identify a clustering configuration (hyperparameter values) that results in more stable clustering results and finally further optimize that configuration through feature selection.

The [fine food reviews example](examples/fine_food_reviews.ipynb) shows how to integrate cutting-edge models (embedding, `4o-mini`) from OpenAI and `KMeans` from `scikit-learn` with `crowflow` to extract meaningful insights from robust clusters and generate informative labels based on these insights.

## License

This package is released under the MIT License.

Developed by Rafael Kollyfas (rk720@cam.ac.uk), Core Bioinformatics (Mohorianu Lab) group, University of Cambridge. February 2025.