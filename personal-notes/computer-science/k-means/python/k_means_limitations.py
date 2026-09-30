"""
K-Means Limitations — Computational Implementation.

NumPy: synthetic data and explicit Lloyd's algorithm.
Matplotlib: figures displayed, not saved.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import Libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import numpy as np
import matplotlib.pyplot as plt


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# K-Means
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def squared_distances(x, centroids):
    """
    Compute squared Euclidean distances to all centroids.

    Input:
        x: Observation matrix.
        centroids: Cluster centroid matrix.

    Output:
        Squared distance from each observation to each centroid.
    """
    differences = x[:, None, :] - centroids[None, :, :]
    return np.sum(differences**2, axis=2)


def assign_clusters(x, centroids):
    """
    Assign each observation to its nearest centroid.

    Input:
        x: Observation matrix.
        centroids: Cluster centroid matrix.

    Output:
        Cluster assignment for each observation.
    """
    distances = squared_distances(x, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(x, labels, centroids):
    """
    Update centroids from the current cluster assignments.

    Input:
        x: Observation matrix.
        labels: Cluster assignment for each observation.
        centroids: Current cluster centroid matrix.

    Output:
        new_centroids: Updated cluster centroids.
        valid: Whether all clusters contain observations.
    """
    k = len(centroids)
    new_centroids = np.empty_like(centroids)

    for cluster in range(k):
        members = x[labels == cluster]

        if len(members) == 0:
            return centroids.copy(), False

        new_centroids[cluster] = np.mean(members, axis=0)

    return new_centroids, True


def objective_function(x, labels, centroids):
    """
    Compute the K-Means within-cluster sum of squares.

    Input:
        x: Observation matrix.
        labels: Cluster assignment for each observation.
        centroids: Cluster centroid matrix.

    Output:
        Within-cluster sum of squares.
    """
    residuals = x - centroids[labels]
    return np.sum(residuals**2)


def k_means(x, initial_centroids, tolerance=1e-8,
            max_iterations=100):
    """
    Fit K-Means from specified initial centroids.

    Input:
        x: Observation matrix.
        initial_centroids: Initial cluster centroids.
        tolerance: Maximum centroid movement for convergence.
        max_iterations: Maximum number of centroid updates.

    Output:
        labels: Final cluster assignments.
        centroids: Final cluster centroids.
        objective: Final within-cluster sum of squares.
        converged: Whether the convergence criterion was reached.
    """
    centroids = np.asarray(
        initial_centroids, dtype=float
    ).copy()

    converged = False

    for _ in range(max_iterations):
        labels = assign_clusters(x, centroids)

        new_centroids, valid = update_centroids(
            x, labels, centroids
        )

        if not valid:
            break

        movement = np.max(
            np.linalg.norm(new_centroids - centroids, axis=1)
        )

        centroids = new_centroids

        if movement <= tolerance:
            converged = True
            break

    labels = assign_clusters(x, centroids)
    objective = objective_function(x, labels, centroids)

    return labels, centroids, objective, converged


def fit_best_k_means(x, k, n_init, rng):
    """
    Fit K-Means repeatedly and retain the smallest objective.

    Input:
        x: Observation matrix.
        k: Number of clusters.
        n_init: Number of random initialisations.
        rng: NumPy random number generator.

    Output:
        best_labels: Cluster assignments from the best run.
        best_centroids: Centroids from the best run.
        best_objective: Smallest final objective.
    """
    best_labels = None
    best_centroids = None
    best_objective = np.inf

    for _ in range(n_init):
        indices = rng.choice(
            len(x), size=k, replace=False
        )
        initial_centroids = x[indices].copy()

        labels, centroids, objective, converged = k_means(
            x, initial_centroids
        )

        if converged and objective < best_objective:
            best_labels = labels.copy()
            best_centroids = centroids.copy()
            best_objective = objective

    if best_labels is None:
        raise RuntimeError(
            "No valid K-Means solution was found."
        )

    return best_labels, best_centroids, best_objective


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Standardisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def standardise(x):
    """
    Standardise features to zero mean and unit variance.

    Input:
        x: Observation matrix.

    Output:
        z: Standardised observation matrix.
        mean: Feature means.
        std: Feature standard deviations.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1)

    if np.any(std == 0):
        raise ValueError(
            "Cannot standardise a constant feature."
        )

    z = (x - mean) / std
    return z, mean, std


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Synthetic Experiments
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def generate_scale_data(n, rng):
    """
    Generate data for the feature-scale experiment.

    Input:
        n: Number of observations per generating group.
        rng: NumPy random number generator.

    Output:
        x: Synthetic observations.
        true_labels: Generating group labels.
    """
    group_1 = rng.multivariate_normal(
        [-2.0, 0.0],
        [[0.5, 0.0], [0.0, 1.0]],
        size=n
    )

    group_2 = rng.multivariate_normal(
        [2.0, 0.0],
        [[0.5, 0.0], [0.0, 1.0]],
        size=n
    )

    x = np.vstack((group_1, group_2))
    true_labels = np.concatenate((
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int),
    ))

    x[:, 1] *= 20.0

    order = rng.permutation(len(x))
    return x[order], true_labels[order]


def generate_outlier_data(n, rng):
    """
    Generate compact groups with an extreme observation.

    Input:
        n: Number of observations per main group.
        rng: NumPy random number generator.

    Output:
        x_clean: Observations without the outlier.
        x_outlier: Observations including the outlier.
    """
    group_1 = rng.multivariate_normal(
        [-2.5, 0.0],
        [[0.5, 0.0], [0.0, 0.5]],
        size=n
    )

    group_2 = rng.multivariate_normal(
        [2.5, 0.0],
        [[0.5, 0.0], [0.0, 0.5]],
        size=n
    )

    x_clean = np.vstack((group_1, group_2))

    outlier = np.array([[10.0, 8.0]])
    x_outlier = np.vstack((x_clean, outlier))

    return x_clean, x_outlier


def generate_non_spherical_data(n, rng):
    """
    Generate two curved non-spherical groups.

    Input:
        n: Number of observations per group.
        rng: NumPy random number generator.

    Output:
        x: Synthetic observations.
        true_labels: Generating group labels.
    """
    theta_1 = rng.uniform(0.0, np.pi, size=n)
    theta_2 = rng.uniform(0.0, np.pi, size=n)

    group_1 = np.column_stack((
        np.cos(theta_1),
        np.sin(theta_1),
    ))

    group_2 = np.column_stack((
        1.0 - np.cos(theta_2),
        0.5 - np.sin(theta_2),
    ))

    group_1 += rng.normal(
        0.0, 0.06, size=group_1.shape
    )
    group_2 += rng.normal(
        0.0, 0.06, size=group_2.shape
    )

    x = np.vstack((group_1, group_2))
    true_labels = np.concatenate((
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int),
    ))

    order = rng.permutation(len(x))
    return x[order], true_labels[order]


def generate_unequal_data(rng):
    """
    Generate groups with unequal sizes and dispersions.

    Input:
        rng: NumPy random number generator.

    Output:
        x: Synthetic observations.
        true_labels: Generating group labels.
    """
    large_group = rng.multivariate_normal(
        [0.0, 0.0],
        [[2.2, 0.0], [0.0, 2.2]],
        size=500
    )

    small_group_1 = rng.multivariate_normal(
        [4.0, 3.5],
        [[0.20, 0.0], [0.0, 0.20]],
        size=60
    )

    small_group_2 = rng.multivariate_normal(
        [-4.0, 3.5],
        [[0.20, 0.0], [0.0, 0.20]],
        size=40
    )

    x = np.vstack((
        large_group,
        small_group_1,
        small_group_2,
    ))

    true_labels = np.concatenate((
        np.zeros(len(large_group), dtype=int),
        np.ones(len(small_group_1), dtype=int),
        np.full(len(small_group_2), 2, dtype=int),
    ))

    order = rng.permutation(len(x))
    return x[order], true_labels[order]


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Comparison Measures
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def adjusted_rand_index(labels_a, labels_b):
    """
    Compute the Adjusted Rand Index between two partitions.

    Input:
        labels_a: Labels for the first partition.
        labels_b: Labels for the second partition.

    Output:
        Adjusted Rand Index.
    """
    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)

    contingency = np.zeros(
        (len(classes_a), len(classes_b)),
        dtype=int
    )

    for i, class_a in enumerate(classes_a):
        for j, class_b in enumerate(classes_b):
            contingency[i, j] = np.sum(
                (labels_a == class_a)
                & (labels_b == class_b)
            )

    def combinations_2(values):
        return values * (values - 1) / 2

    sum_cells = np.sum(
        combinations_2(contingency)
    )

    row_sums = np.sum(contingency, axis=1)
    column_sums = np.sum(contingency, axis=0)

    sum_rows = np.sum(
        combinations_2(row_sums)
    )
    sum_columns = np.sum(
        combinations_2(column_sums)
    )

    n = len(labels_a)
    total_pairs = n * (n - 1) / 2

    expected = (
        sum_rows * sum_columns / total_pairs
    )

    maximum = 0.5 * (
        sum_rows + sum_columns
    )

    denominator = maximum - expected

    if denominator == 0:
        return 1.0

    return (sum_cells - expected) / denominator


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_partition(x, labels, centroids, title):
    """
    Plot a K-Means partition and its centroids.

    Input:
        x: Observation matrix.
        labels: Cluster assignments.
        centroids: Cluster centroids.
        title: Figure title.

    Output:
        Displays the clustering.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in np.unique(labels):
        members = x[labels == cluster]

        ax.scatter(
            members[:, 0],
            members[:, 1],
            s=20,
            alpha=0.55,
            label=f"Cluster {cluster + 1}"
        )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=150,
        edgecolor="black",
        linewidth=1,
        label="Centroids"
    )

    ax.set(
        title=title,
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_true_structure(x, true_labels, title):
    """
    Plot the generating structure of synthetic observations.

    Input:
        x: Observation matrix.
        true_labels: Generating group labels.
        title: Figure title.

    Output:
        Displays the generating groups.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for group in np.unique(true_labels):
        members = x[true_labels == group]

        ax.scatter(
            members[:, 0],
            members[:, 1],
            s=20,
            alpha=0.55,
            label=f"Group {group + 1}"
        )

    ax.set(
        title=title,
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_outlier_effect(
    x_clean,
    clean_labels,
    clean_centroids,
    x_outlier,
    outlier_labels,
    outlier_centroids,
):
    """
    Compare K-Means centroids before and after adding an outlier.

    Input:
        x_clean: Dataset without the outlier.
        clean_labels: Clusters without the outlier.
        clean_centroids: Centroids without the outlier.
        x_outlier: Dataset including the outlier.
        outlier_labels: Clusters including the outlier.
        outlier_centroids: Centroids including the outlier.

    Output:
        Displays separate before-and-after figures.
    """
    plot_partition(
        x_clean,
        clean_labels,
        clean_centroids,
        "K-Means Without Outlier"
    )

    plot_partition(
        x_outlier,
        outlier_labels,
        outlier_centroids,
        "K-Means With Outlier"
    )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Demonstrate selected geometric limitations of K-Means.

    Input:
        None.

    Output:
        Prints experimental results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n_init = 20
    separator = "<>" * 36

    print(separator)
    print("K-Means Limitations")
    print(separator)

    # Feature Scale

    x_scale, true_scale = generate_scale_data(
        200, rng
    )

    (
        labels_raw,
        centroids_raw,
        objective_raw,
    ) = fit_best_k_means(
        x_scale, 2, n_init, rng
    )

    z_scale, mean_scale, std_scale = standardise(
        x_scale
    )

    (
        labels_scaled,
        centroids_scaled,
        objective_scaled,
    ) = fit_best_k_means(
        z_scale, 2, n_init, rng
    )

    ari_raw = adjusted_rand_index(
        true_scale, labels_raw
    )

    ari_scaled = adjusted_rand_index(
        true_scale, labels_scaled
    )

    print()
    print(separator)
    print("Feature Scale")
    print(separator)
    print()

    print(
        f"Original feature std.: "
        f"{np.array2string(std_scale, precision=4)}"
    )
    print(f"ARI before scaling:    {ari_raw:.6f}")
    print(f"ARI after scaling:     {ari_scaled:.6f}")
    print(f"Raw-space WCSS:        {objective_raw:.6f}")
    print(f"Scaled-space WCSS:     {objective_scaled:.6f}")

    plot_true_structure(
        x_scale,
        true_scale,
        "Generating Structure — Unequal Feature Scales"
    )

    plot_partition(
        x_scale,
        labels_raw,
        centroids_raw,
        "K-Means Before Standardisation"
    )

    centroids_original_scale = (
        centroids_scaled * std_scale + mean_scale
    )

    plot_partition(
        x_scale,
        labels_scaled,
        centroids_original_scale,
        "K-Means After Standardisation"
    )

    # Outliers

    x_clean, x_outlier = generate_outlier_data(
        200, rng
    )

    (
        clean_labels,
        clean_centroids,
        clean_objective,
    ) = fit_best_k_means(
        x_clean, 2, n_init, rng
    )

    (
        outlier_labels,
        outlier_centroids,
        outlier_objective,
    ) = fit_best_k_means(
        x_outlier, 2, n_init, rng
    )

    centroid_shift = np.min(
        np.linalg.norm(
            clean_centroids[:, None, :]
            - outlier_centroids[None, :, :],
            axis=2
        ),
        axis=1
    )

    print()
    print(separator)
    print("Outlier Sensitivity")
    print(separator)
    print()

    print(f"WCSS without outlier:  {clean_objective:.6f}")
    print(f"WCSS with outlier:     {outlier_objective:.6f}")
    print(
        f"Centroid shifts:       "
        f"{np.array2string(centroid_shift, precision=4)}"
    )

    plot_outlier_effect(
        x_clean,
        clean_labels,
        clean_centroids,
        x_outlier,
        outlier_labels,
        outlier_centroids,
    )

    # Non-Spherical Structure

    x_curved, true_curved = generate_non_spherical_data(
        250, rng
    )

    (
        curved_labels,
        curved_centroids,
        curved_objective,
    ) = fit_best_k_means(
        x_curved, 2, n_init, rng
    )

    ari_curved = adjusted_rand_index(
        true_curved, curved_labels
    )

    print()
    print(separator)
    print("Non-Spherical Structure")
    print(separator)
    print()

    print(f"Generating groups:     2")
    print(f"K-Means clusters:      2")
    print(f"WCSS:                  {curved_objective:.6f}")
    print(f"ARI:                   {ari_curved:.6f}")

    plot_true_structure(
        x_curved,
        true_curved,
        "Generating Non-Spherical Structure"
    )

    plot_partition(
        x_curved,
        curved_labels,
        curved_centroids,
        "K-Means on Non-Spherical Structure"
    )

    # Unequal Sizes and Dispersions

    x_unequal, true_unequal = generate_unequal_data(
        rng
    )

    (
        unequal_labels,
        unequal_centroids,
        unequal_objective,
    ) = fit_best_k_means(
        x_unequal, 3, n_init, rng
    )

    true_sizes = np.bincount(true_unequal)
    cluster_sizes = np.bincount(
        unequal_labels, minlength=3
    )

    ari_unequal = adjusted_rand_index(
        true_unequal, unequal_labels
    )

    print()
    print(separator)
    print("Unequal Sizes and Dispersions")
    print(separator)
    print()

    print(
        f"Generating sizes:      "
        f"{np.array2string(true_sizes)}"
    )
    print(
        f"K-Means sizes:         "
        f"{np.array2string(cluster_sizes)}"
    )
    print(f"WCSS:                  {unequal_objective:.6f}")
    print(f"ARI:                   {ari_unequal:.6f}")

    plot_true_structure(
        x_unequal,
        true_unequal,
        "Generating Groups — Unequal Sizes and Dispersions"
    )

    plot_partition(
        x_unequal,
        unequal_labels,
        unequal_centroids,
        "K-Means — Unequal Sizes and Dispersions"
    )


if __name__ == "__main__":
    main()