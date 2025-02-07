from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    geom_point,
    geom_line,
    theme_classic,
    labs,
    ggtitle,
    scale_color_cmap,
    geom_hline,
    annotate,
    ylim,
    geom_tile,
    scale_fill_cmap,
    theme_minimal,
    theme,
    element_text
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_scatter_ecc(
    df,
    x_col,
    y_col,
    color_col,
    title="Scatter Plot - ECC",
    alpha=0.7,
    size=2,
    cmap="viridis",
):
    """
    Generate a scatter plot with color-coded points. Intended to be used for visualising the 
    stability of invidual points using element-centric consistency (i.e., color_col = ecc).

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    x_col : str
        The name of the column to plot on the x-axis.
    y_col : str
        The name of the column to plot on the y-axis.
    color_col : str
        The name of the column to use for coloring the points.
    title : str, optional (default="Scatter Plot - ECC")
        The title of the plot.
    alpha : float, optional (default=0.7)
        The transparency level for the points.
    size : float, optional (default=2)
        The size of the points.
    cmap : str, optional (default="viridis")
        The color map used for the points.

    Returns
    -------
    plotnine.ggplot
        The generated scatter plot.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import pandas as pd
    >>> from umap import UMAP
    >>> from sklearn.cluster import KMeans
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
    >>> reducer = UMAP(random_state=42)
    >>> umap_embedding = reducer.fit_transform(df)
    >>> umap_df = pd.DataFrame(data=umap_embedding, columns=['UMAP_1', 'UMAP_2'])
    >>> repeated_clustering_func = StochasticClusteringRunner(KMeans, "random_state", n_runs=30, verbose=False)
    >>> results_repeated_clustering = repeated_clustering_func.run(df)
    >>> umap_df['ecc'] = results_repeated_clustering['ecc']
    >>> plot_scatter_ecc(umap_df, "UMAP_1", "UMAP_2", "ecc")
    """
    plot = (
        ggplot(df, aes(x=x_col, y=y_col, color=color_col))
        + geom_point(size=size, alpha=alpha)
        + theme_classic()
        + labs(x=x_col, y=y_col, color=color_col)
        + scale_color_cmap(cmap_name=cmap)
        + ggtitle(title)
    )
    return plot.draw()


def plot_optimization_of_parameter_boxplot(parameter_optimizer_results_df):
    """
    Generate a boxplot visualization of element-centric consistency (ECC) values for different parameters and values tested. 
    Expected input is the DataFrame output of the run function of ParameterOptimizer.

    Parameters
    ----------
    parameter_optimizer_results_df : pandas.DataFrame
        A DataFrame with at least two columns:
        - "param": The name of parameters and their values (e.g., paramA_2) tested during optimization.
        - "ecc": A list of ECC values corresponding to each parameter setting.

    Returns
    -------
    plotnine.ggplot
        The generated boxplot visualization.

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
    >>> plot_optimization_of_parameter_boxplot(results_df)
    """
    # Convert to long format
    df_long = parameter_optimizer_results_df.copy()
    df_long = df_long.explode("ecc")

    # Convert ECC column to numeric
    df_long["ecc"] = pd.to_numeric(df_long["ecc"], errors="coerce")

    plot = (
        ggplot(df_long, aes(x="param", y="ecc"))
        + geom_boxplot(fill="lightblue", color="black")
        + theme_classic()
        + labs(x="Params", y="ECC")
        + ggtitle("Parameter Optimization - Boxplot Distribution")
    )

    return plot.draw()


def plot_heatmap(
    parameter_search_results_df,
    key1=None,
    key2=None,
    agg_func=np.median,
    fixed_params=None,
):
    """
    Plot a heatmap of median element-centric consistency (ECC) scores for combinations of two parameters. 
    Expected input is the DataFrame output of the run function of ParameterSearcher.

    Parameters
    ----------
    parameter_search_results_df : DataFrame
        A data frame containing a 'params' column (list of parameter settings)
        and a 'median_ecc' column.
    key1 : str, optional
        The first parameter for the heatmap. If None, the first parameter in the data is used.
    key2 : str, optional
        The second parameter for the heatmap. If None, the second parameter in the data is used.
    agg_func : function, optional
        Aggregation function (e.g., np.median, np.mean). Default is np.median.
    fixed_params : dict, optional
        A dictionary of parameters to fix at specific values.

    Returns
    -------
    plotnine.ggplot
        A ggplot object representing the heatmap.

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
    >>> plot_heatmap(results_df)
    """
    if not {"params", "median_ecc"}.issubset(parameter_search_results_df.columns):
        raise ValueError("The data frame must contain 'params' and 'median_ecc' columns.")

    first_params = parameter_search_results_df["params"].iloc[0]
    param_keys = list(first_params.keys())

    # Select keys for the heatmap
    if len(param_keys) >= 2:
        if key1 is None:
            key1 = param_keys[0]
        if key2 is None:
            key2 = param_keys[1]
        print(f"Selected keys for visualization: {key1}, {key2}")
    else:
        raise ValueError("You must have at least two parameters to create a heatmap.")

    # Verify that key1 and key2 are in the parameter keys
    if key1 not in param_keys or key2 not in param_keys:
        raise ValueError("Specify valid 'key1' and 'key2' parameters for visualization.")

    print("Creating DataFrame from parameter search results...")
    # Create DataFrame from list of parameter dictionaries and add median_ecc column
    params_df = pd.DataFrame(parameter_search_results_df["params"].tolist())
    params_df["median_ecc"] = parameter_search_results_df["median_ecc"].astype(float)

    # Apply fixed parameters if provided
    if fixed_params is not None:
        print(f"Applying filters for fixed parameters: {', '.join(fixed_params.keys())}")
        for param, value in fixed_params.items():
            if param in params_df.columns:
                before_filter = params_df.shape[0]
                params_df = params_df[params_df[param] == value]
                after_filter = params_df.shape[0]
                print(f"Filtered {param} to {value}. Rows: {before_filter} -> {after_filter}")
            else:
                print(f"Warning: Parameter '{param}' not in dataset. Skipping.")

    # Check for duplicate combinations of key1 and key2
    unique_combinations = params_df[[key1, key2]].drop_duplicates()
    if params_df.shape[0] != unique_combinations.shape[0]:
        print("Duplicate combinations found. Aggregating values...")
        heatmap_data = (
            params_df.groupby([key1, key2])["median_ecc"]
            .agg(agg_func)
            .reset_index()
        )
        print(f"Aggregation with '{agg_func.__name__}' done.")
    else:
        print("No duplicates found. Proceeding without aggregation.")
        heatmap_data = params_df[[key1, key2, "median_ecc"]]

    # Convert key1 and key2 to categorical if they are not numeric
    if not np.issubdtype(heatmap_data[key1].dtype, np.number):
        heatmap_data[key1] = heatmap_data[key1].astype("category")
    if not np.issubdtype(heatmap_data[key2].dtype, np.number):
        heatmap_data[key2] = heatmap_data[key2].astype("category")

    plot = (
        ggplot(heatmap_data, aes(x=key2, y=key1, fill="median_ecc"))
        + geom_tile(color="white")
        + scale_fill_cmap(name="ECC")
        + theme_minimal()
        + labs(
            title="Heatmap of Median ECC",
            x=key2,
            y=key1
        )
        + theme(
            axis_text_x=element_text(rotation=45, ha="right")
        )
    )

    return plot.draw()


def plot_ga_fitness_evolution(ga_fs_history):
    """
    Plot the best fitness evolution over generations. 
    Expected input is the history output of the run function of GeneticAlgorithmFeatureSelector.

    Parameters
    ----------
    ga_fs_history : pandas.DataFrame
        DataFrame containing the history of the feature selection GA from the GeneticAlgorithmFeatureSelector.

    Returns
    -------
    plotnine.ggplot
        The generated plot
    
    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.normal(size=(200, 10)), columns=[f"feature_{i+1}" for i in range(10)])
    >>> ga_fs = GeneticAlgorithmFeatureSelector(KMeans, 'random_state', verbose=True, n_generations_no_change=5)
    >>> ga_results = ga_fs.run(df)
    >>> plot_ga_fitness_evolution(ga_results['history'])
    """
    plot = (
        ggplot(ga_fs_history, aes(x="generation", y="best_fitness"))
        + geom_line(size=1)
        + geom_point(size=3)
        + theme_classic()
        + labs(x="Generation", y="Best Fitness")
        + ggtitle("Feature Selection GA - Best Fitness Evolution")
    )
    return plot.draw()


def plot_kfold_scores(kfolds_results, agg_func=np.median):
    """
    Plot aggregated element-centric similarity (ECS) scores across k-folds. 
    Expected input is the history output of the run function of GeneticAlgorithmFeatureSelector.

    Parameters
    ----------
    kfolds_results : dict
        Dictionary containing k-fold results from KFoldClusteringValidator
    agg_func : callable, optional (default=np.median)
        Function to aggregate scores (e.g., np.median, np.mean)

    Returns
    -------
    plotnine.ggplot
        The generated plot

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
    >>> plot_kfold_scores(kfolds_robustness_results)
    """
    data = []
    for fold, results in kfolds_results.items():
        score = agg_func(results["el_score_vector"])
        data.append({"Fold": fold, "Score": score})

    df = pd.DataFrame(data)

    # Sort folds by fold number
    def fold_sort_key(fold_name):
        return int(fold_name.split("_")[1])

    df["Fold"] = pd.Categorical(
        df["Fold"],
        categories=sorted(df["Fold"].unique(), key=fold_sort_key),
        ordered=True,
    )

    overall_score = agg_func(df["Score"])

    plot = (
        ggplot(df, aes(x="Fold", y="Score", group=1))
        + geom_line(size=1)
        + geom_point(size=3)
        + geom_hline(yintercept=overall_score, linetype="dashed", color="red", size=1)
        + theme_classic()
        + labs(
            x="Fold",
            y="Element Similarity Score",
            title=f"{agg_func.__name__.capitalize()} Element Centric Similarity Scores Across K-Folds",
        )
        + annotate(
            "text",
            x=df["Fold"].max(),
            y=min(overall_score - 0.1, 1.0),
            label=f"Overall {agg_func.__name__}: {overall_score:.3f}",
            color="red",
            ha="right",
        )
        + ylim(0, 1.0)  # Set the y-axis limits to ensure the maximum is capped at 1.0
    )

    return plot.draw()
