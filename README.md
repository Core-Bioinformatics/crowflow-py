# Crow

Crow is a Python package designed for assessing clustering stability through repeated stochastic clustering. It is compatible with any clustering algorithm that outputs labels or implements a fit or fit_predict method, provided it includes stochasticity (i.e., allows setting a seed or random_state). By running clustering multiple times with different seeds, Crow quantifies clustering consistency using element-centric similarity (ECS) and element-centric consistency (ECC), offering insights into the robustness and reproducibility of cluster assignments. The package enables users to optimize feature subsets, fine-tune clustering parameters, and evaluate clustering robustness against perturbations.

Crow generalizes the ClustAssessPy package, which focuses on parameter selection for community-detection clustering in single-cell analysis. It extends this approach to any clustering task, enabling a data-driven identification of robust and reproducible clustering solutions across diverse applications.

## Class Summaries
- `StochasticClusteringRunner`: Runs a stochastic clustering algorithm multiple times with different random seeds and evaluates the stability of results using ECC. It identifies in an element-wise precision the stability of clustering results and provides majority voting label.
- `GeneticAlgorithmFeatureSelector`: Uses a genetic algorithm to iteratively optimize feature selection for clustering stability. It repeatedly applies stochastic clustering with different feature subsets and evaluates stability using ECC. The algorithm evolves through selection, crossover, and mutation, converging on the feature set that maximizes clustering robustness.
- `ParameterOptimizer`: Systematically tunes each hyperparameter separately by performing repeated clustering and evaluating stability using ECC. 
- `ParameterSearcher`: Evaluates all possible combinations (exhaustive grid search) of specified parameters running repeated clustering and computing ECC for each combination. Purpose is to find configuration (set of hyperparameter values) that provide the most stable clustering results.
- `KFoldClusteringValidator`: Evaluates how stable clustering assignments remain across different data partitions by comparing clustering results on k-fold subsets with those  from the full dataset. ECS is used to quantify consistency between fold-level clustering and the baseline (full dataset).
- `PerturbationRobustnessTester`: Tests how stable clustering results are when features are altered/perturbed. The user must provide a perturbation function, which modifies the dataset before clustering is re-run. Stability is assessed using Element-Centric Similarity (ECS) between the baseline clustering and perturbation-induced clusterings.

## Installation

Crow requires Python 3.7 or newer.

### Dependencies

- numpy
- matplotlib
- scikit-learn
- seaborn
- plotnine
- ClustAssessPy


### User Installation

We recommend that you download `crow` on a virtual environment (venv or Conda).

```sh
pip install crow
```

## Tutorials

The package can be applied to any clustering task (as long as the clustering algorithm used is stochastic). 

In the [cuomo example](examples/cuomo_application.ipynb) we show how to use `crow` with GaussianMixture from `scikit-learn` to initially assess clustering stability of the default parameter values. We then attempt to identify a clustering configuration (hyperparameter values) that would result in more stable clustering results and finally further optimise that configuration by feature selection. 

The [fine food reviews example](examples/fine_food_reviews.ipynb) shows how to integrate cutting-edge models (embedding, 4o-mini) from OpenAI and KMeans from `scikit-learn` with `crow`to extract meaningful insights from the identified robust clusters, and generate informative labels based on these insights.

## License

MIT

Developed by Rafael Kollyfas (rk720@cam.ac.uk), Core Bioinformatics (Mohorianu Lab) group, University of Cambridge. February 2025.
