from ..clustering.repeated_stochastic_clustering import StochasticClusteringRunner
import pandas as pd
import numpy as np


class GeneticAlgorithmFeatureSelector:
    """
    Selects the most stable subset of features using a genetic algorithm.

    This class uses a genetic algorithm to iteratively optimize feature selection for
    clustering stability. It repeatedly applies stochastic clustering with different
    feature subsets and evaluates stability using Element-Centric Consistency (ECC).
    The algorithm evolves through selection, crossover, and mutation, converging on
    the feature set that maximizes clustering robustness.

    Parameters
    ----------
    clustering_algo : callable
        Clustering function or class (e.g., KMeans from scikit-learn).
    parameter_name_seed : str
        Name of the parameter used to set the random seed.
    n_runs : int, optional (default=30)
        Number of repeated clustering runs per feature subset.
    population_size : int, optional (default=20)
        Number of feature subsets in each generation.
    generations : int, optional (default=50)
        Number of generations to evolve.
    tournament_selection_k : int, optional (default=3)
        Tournament size for parent selection.
    mutation_rate : float, optional (default=0.01)
        Probability of flipping a feature selection bit in offspring.
    crossover_rate : float, optional (default=0.8)
        Probability of performing crossover between two parents.
    elite_size : int, optional (default=2)
        Number of top-performing subsets preserved per generation.
    n_generations_no_change : int, optional (default=10)
        Early stopping criterion: generations without improvement.
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
    >>> ga_fs = GeneticAlgorithmFeatureSelector(KMeans, 'random_state', verbose=True, n_generations_no_change=5)
    >>> ga_results = ga_fs.run(df)
    >>> print("Best Features:", ga_results["best_features"])
    >>> print("Best ECC Score:", ga_results["best_ecc"])
    """

    def __init__(
        self,
        clustering_algo,
        parameter_name_seed,
        n_runs=30,
        population_size=20,
        generations=50,
        tournament_selection_k=3,
        mutation_rate=0.01,
        crossover_rate=0.8,
        elite_size=2,
        n_generations_no_change=10,
        verbose=False,
        labels_name=None,
        **kwargs,
    ):
        np.random.seed(42)
        self.clustering_algo = clustering_algo
        self.parameter_name_seed = parameter_name_seed
        self.n_runs = n_runs
        self.population_size = population_size
        self.generations = generations
        self.tournament_selection_k = tournament_selection_k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.n_generations_no_change = n_generations_no_change
        self.verbose = verbose
        self.labels_name = labels_name
        self.kwargs = kwargs

    def run(self, data):
        """
        Runs the genetic algorithm for feature selection.

        The method initializes a population of feature subsets, evaluates their stability
        via repeated clustering, and evolves them using genetic operators (selection,
        crossover, mutation). The best subset maximizes Element-Centric Consistency (ECC),
        ensuring robust clustering.

        Parameters
        ----------
        data : array-like or DataFrame
            Input dataset of shape (n_samples, n_features).

        Returns
        -------
        dict
            - 'best_features': Names of the most stable feature subset.
            - 'best_ecc': Highest median ECC achieved during evolution.
            - 'history': DataFrame tracking best fitness over generations.
            - 'best_fitness_scr_result': Clustering results for the optimal feature set.
        """
        # Convert to numpy if DataFrame
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns
            data = data.values
        else:
            feature_names = None

        n_features = data.shape[1]

        # Initialize population: each candidate is a binary vector
        population = np.random.randint(2, size=(self.population_size, n_features))

        def evaluate_fitness(candidate):
            # candidate is a binary mask
            chosen_features = np.where(candidate == 1)[0]
            if len(chosen_features) == 0:
                return -np.inf, None
            subset_data = data[:, chosen_features]

            runner = StochasticClusteringRunner(
                self.clustering_algo,
                self.parameter_name_seed,
                n_runs=self.n_runs,
                verbose=False,
                labels_name=self.labels_name,
                **self.kwargs,
            )
            results = runner.run(subset_data)
            ecc = results["ecc"]
            median_ecc = np.median(ecc)
            return median_ecc, results

        # Evaluate initial population
        fitness_scores = np.empty(self.population_size)
        scr_results_list = [None] * self.population_size
        for idx in range(self.population_size):
            fitness, scr_result = evaluate_fitness(population[idx])
            fitness_scores[idx] = fitness
            scr_results_list[idx] = scr_result

        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_candidate = population[best_idx].copy()
        best_fitness_scr_result = scr_results_list[best_idx]

        history = [{"generation": 0, "best_fitness": best_fitness}]
        if self.verbose:
            print(f"Gen 0 - Best ECC: {best_fitness:.4f}")

        iters_no_improvement = 0
        for gen in range(1, self.generations + 1):
            if iters_no_improvement >= self.n_generations_no_change:
                break

            def tournament_selection(k):
                selected = np.random.choice(self.population_size, k, replace=False)
                best = selected[np.argmax(fitness_scores[selected])]
                return population[best]

            # Create new population
            new_population = []

            # Elitism: keep best candidates
            elite_indices = np.argsort(fitness_scores)[-self.elite_size :]
            elites = population[elite_indices]
            for e in elites:
                new_population.append(e.copy())

            # Fill the rest of the population
            while len(new_population) < self.population_size:
                parent1 = tournament_selection(self.tournament_selection_k)
                parent2 = tournament_selection(self.tournament_selection_k)

                # Crossover
                if np.random.rand() < self.crossover_rate:
                    point = np.random.randint(1, n_features)
                    offspring1 = np.concatenate([parent1[:point], parent2[point:]])
                    offspring2 = np.concatenate([parent2[:point], parent1[point:]])
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                for offspring in [offspring1, offspring2]:
                    for i in range(n_features):
                        if np.random.rand() < self.mutation_rate:
                            offspring[i] = 1 - offspring[i]

                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)

            population = np.array(new_population)
            fitness_scores = np.empty(self.population_size)
            generation_scr_results_list = [None] * self.population_size
            for idx in range(self.population_size):
                fitness, scr_result = evaluate_fitness(population[idx])
                fitness_scores[idx] = fitness
                generation_scr_results_list[idx] = scr_result

            generation_best_idx = np.argmax(fitness_scores)
            generation_best_fitness = fitness_scores[generation_best_idx]

            # Update global best if improved
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_candidate = population[generation_best_idx].copy()
                best_fitness_scr_result = generation_scr_results_list[
                    generation_best_idx
                ]
                iters_no_improvement = 0
            else:
                iters_no_improvement += 1

            history.append({"generation": gen, "best_fitness": best_fitness})
            if self.verbose:
                print(f"Gen {gen} - Best ECC: {best_fitness:.4f}")

        history_df = pd.DataFrame(history)
        chosen_features = np.where(best_candidate == 1)[0]

        if feature_names is not None:
            chosen_features = feature_names[chosen_features]

        return {
            "best_features": chosen_features,
            "best_ecc": best_fitness,
            "history": history_df,
            "best_fitness_scr_result": best_fitness_scr_result,
        }
