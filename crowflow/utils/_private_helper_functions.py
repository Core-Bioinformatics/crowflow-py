import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter

def _get_clustering_labels(data, clustering_algo, algo_params, labels_name=None):
    if hasattr(clustering_algo, "fit_predict"):
        # For scikit-learn-style algorithms with fit_predict method
        model = clustering_algo(**algo_params)
        return model.fit_predict(data)
    elif hasattr(clustering_algo, "fit"):
        # For scikit-learn-style algorithms with fit method
        model = clustering_algo(**algo_params)
        model.fit(data)
        return model.labels_
    elif callable(clustering_algo):
        result = clustering_algo(data, **algo_params)  # Run clustering

        # Extract labels from Leiden or other clustering outputs
        if labels_name is None:
            return list(result)  # Assume the function directly returns labels
        elif isinstance(labels_name, str):
            if hasattr(result, labels_name):
                return getattr(result, labels_name)  # Extract specified attribute
            else:
                raise ValueError(
                    f"Clustering result does not have attribute `{labels_name}`."
                )
        else:
            raise ValueError(
                "`labels_name` must be a string or None."
            )
    else:
        raise ValueError(
            "Provided clustering_algo must be callable or have a 'fit' or 'fit_predict' method."
        )



def _reconcile_partitions_and_majority_voting(partition_frequencies):
    if not partition_frequencies:
        raise ValueError("partition_frequencies is empty.")

    # Convert all partition keys to tuples of standard ints
    converted_partition_frequencies = {}
    for partition, freq in partition_frequencies.items():
        # Ensure partition is a tuple and all elements are standard ints
        if not isinstance(partition, tuple):
            raise TypeError(f"Partition keys must be tuples. Got {type(partition)}")
        converted_partition = tuple(int(x) for x in partition)
        converted_partition_frequencies[converted_partition] = freq

    # Sort partitions by frequency in descending order
    sorted_partitions = sorted(
        converted_partition_frequencies.items(), key=lambda item: item[1], reverse=True
    )
    sorted_unique_partitions = [partition for partition, freq in sorted_partitions]

    # Reference partition is the most frequent one
    reference_partition = sorted_unique_partitions[0]
    reconciled_partitions = [reference_partition]
    mapping_dict = {reference_partition: reference_partition}

    unique_ref = np.unique(reference_partition)

    for current_partition in sorted_unique_partitions[1:]:
        unique_current = np.unique(current_partition)

        # Create cost matrix based on 1 - Jaccard Similarity Index
        cost_matrix = np.zeros((len(unique_current), len(unique_ref)), dtype=float)

        for ix, c1 in enumerate(unique_current):
            mask1 = np.array(current_partition) == c1
            sum1 = np.sum(mask1)
            for jx, c2 in enumerate(unique_ref):
                mask2 = np.array(reference_partition) == c2
                intersection = np.sum(mask1 & mask2)
                union = sum1 + np.sum(mask2) - intersection
                jaccard_index = intersection / union if union != 0 else 0.0
                cost_matrix[ix, jx] = 1 - jaccard_index  # Cost

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create mapping from current_partition to reference_partition
        mapping_from = unique_current[row_ind]
        mapping_to = unique_ref[col_ind]

        # Adjust the current_partition labels based on the mapping
        adjusted_partition = np.copy(current_partition)
        for src, dst in zip(mapping_from, mapping_to):
            adjusted_partition[current_partition == src] = dst

        adjusted_partition = tuple(int(x) for x in adjusted_partition)
        reconciled_partitions.append(adjusted_partition)
        mapping_dict[current_partition] = adjusted_partition

    labels_list = []
    for partition, freq in sorted_partitions:
        partition = tuple(int(x) for x in partition)
        reconciled_partition = mapping_dict.get(partition)
        if reconciled_partition is None:
            raise ValueError(f"Partition {partition} not found in mapping_dict.")
        labels_list.extend([reconciled_partition] * freq)

    if not labels_list:
        raise ValueError("No labels to process after reconciliation.")

    stacked_partitions = np.array(labels_list, dtype=int)
    stacked_partitions = stacked_partitions.T

    majority_voting_labels = np.array(
        [
            Counter(sample_labels).most_common(1)[0][0]
            for sample_labels in stacked_partitions
        ]
    )

    return majority_voting_labels, reconciled_partitions
