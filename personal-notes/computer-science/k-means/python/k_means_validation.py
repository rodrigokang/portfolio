"""
K-Means Validation — Computational Implementation.

NumPy: explicit K-Means and internal validation measures.
Matplotlib: figures displayed, not saved.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import Libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import numpy as np
import matplotlib.pyplot as plt


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Synthetic Data
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def generate_data(n_per_cluster, means, covariance, rng):
    """
    Generate synthetic observations from Gaussian clusters.

    Input:
        n_per_cluster: Number of observations in each cluster.
        means: Mean vector for each cluster.
        covariance: Common covariance matrix.
        rng: NumPy random number generator.

    Output:
        x: Synthetic observations.
    """
    observations = []

    for mean in means:
        cluster = rng.multivariate_normal(
            mean, covariance, size=n_per_cluster
        )
        observations.append(cluster)

    x = np.vstack(observations)
    return x[rng.permutation(len(x))]


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Distance and Objective Function
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


def pairwise_distances(x):
    """
    Compute pairwise Euclidean distances between observations.

    Input:
        x: Observation matrix.

    Output:
        Pairwise Euclidean distance matrix.
    """
    differences = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


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


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Lloyd's Algorithm
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

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


def random_centroids(x, k, rng):
    """
    Select distinct observations as initial centroids.

    Input:
        x: Observation matrix.
        k: Number of clusters.
        rng: NumPy random number generator.

    Output:
        Initial cluster centroids.
    """
    indices = rng.choice(len(x), size=k, replace=False)
    return x[indices].copy()


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
        initial_centroids = random_centroids(x, k, rng)

        labels, centroids, objective, converged = k_means(
            x, initial_centroids
        )

        if converged and objective < best_objective:
            best_labels = labels.copy()
            best_centroids = centroids.copy()
            best_objective = objective

    if best_labels is None:
        raise RuntimeError(
            f"No valid K-Means solution found for K={k}."
        )

    return best_labels, best_centroids, best_objective


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Silhouette Coefficient
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def silhouette_coefficients(x, labels):
    """
    Compute the Silhouette coefficient for each observation.

    Input:
        x: Observation matrix.
        labels: Cluster assignment for each observation.

    Output:
        silhouette: Silhouette coefficient for each observation.
    """
    distances = pairwise_distances(x)
    unique_labels = np.unique(labels)
    silhouette = np.zeros(len(x))

    if len(unique_labels) < 2:
        raise ValueError(
            "Silhouette requires at least two clusters."
        )

    for i in range(len(x)):
        own_cluster = labels[i]
        own_mask = labels == own_cluster
        own_mask[i] = False

        if np.sum(own_mask) == 0:
            silhouette[i] = 0.0
            continue

        a_i = np.mean(distances[i, own_mask])

        b_i = np.inf

        for cluster in unique_labels:
            if cluster == own_cluster:
                continue

            other_mask = labels == cluster
            mean_distance = np.mean(
                distances[i, other_mask]
            )
            b_i = min(b_i, mean_distance)

        denominator = max(a_i, b_i)

        if denominator > 0:
            silhouette[i] = (
                (b_i - a_i) / denominator
            )

    return silhouette


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Calinski-Harabasz Index
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def calinski_harabasz_index(x, labels, centroids):
    """
    Compute the Calinski-Harabasz index.

    Input:
        x: Observation matrix.
        labels: Cluster assignment for each observation.
        centroids: Cluster centroid matrix.

    Output:
        Calinski-Harabasz index.
    """
    n = len(x)
    k = len(centroids)

    if k < 2 or k >= n:
        raise ValueError(
            "Calinski-Harabasz requires 1 < K < N."
        )

    overall_mean = np.mean(x, axis=0)
    within = 0.0
    between = 0.0

    for cluster in range(k):
        members = x[labels == cluster]

        within += np.sum(
            (members - centroids[cluster])**2
        )

        between += len(members) * np.sum(
            (centroids[cluster] - overall_mean)**2
        )

    if within <= 0:
        return np.inf

    return (
        (between / (k - 1))
        / (within / (n - k))
    )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Davies-Bouldin Index
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def davies_bouldin_index(x, labels, centroids):
    """
    Compute the Davies-Bouldin index.

    Input:
        x: Observation matrix.
        labels: Cluster assignment for each observation.
        centroids: Cluster centroid matrix.

    Output:
        Davies-Bouldin index.
    """
    k = len(centroids)
    dispersions = np.zeros(k)

    for cluster in range(k):
        members = x[labels == cluster]

        dispersions[cluster] = np.mean(
            np.linalg.norm(
                members - centroids[cluster],
                axis=1
            )
        )

    centroid_distances = np.sqrt(
        squared_distances(centroids, centroids)
    )

    similarities = np.full((k, k), -np.inf)

    for cluster in range(k):
        for other in range(k):
            if cluster == other:
                continue

            distance = centroid_distances[cluster, other]

            if distance <= 0:
                similarities[cluster, other] = np.inf
            else:
                similarities[cluster, other] = (
                    dispersions[cluster]
                    + dispersions[other]
                ) / distance

    worst_similarity = np.max(similarities, axis=1)
    return np.mean(worst_similarity)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Validation Experiment
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def evaluate_clusterings(x, k_values, n_init, rng):
    """
    Evaluate K-Means solutions across several values of K.

    Input:
        x: Observation matrix.
        k_values: Candidate numbers of clusters.
        n_init: Number of initialisations for each K.
        rng: NumPy random number generator.

    Output:
        results: Validation results for each value of K.
    """
    results = []

    for k in k_values:
        labels, centroids, objective = fit_best_k_means(
            x, k, n_init, rng
        )

        silhouette = silhouette_coefficients(x, labels)

        results.append({
            "k": k,
            "labels": labels,
            "centroids": centroids,
            "objective": objective,
            "silhouette": silhouette,
            "mean_silhouette": np.mean(silhouette),
            "calinski_harabasz": calinski_harabasz_index(
                x, labels, centroids
            ),
            "davies_bouldin": davies_bouldin_index(
                x, labels, centroids
            ),
        })

    return results


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_elbow(results):
    """
    Plot within-cluster sum of squares against K.

    Input:
        results: Validation results for each value of K.

    Output:
        Displays the Elbow diagnostic.
    """
    k_values = np.array([
        result["k"] for result in results
    ])

    objectives = np.array([
        result["objective"] for result in results
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        k_values,
        objectives,
        marker="o",
        linewidth=1.5
    )

    ax.set(
        title="Elbow Diagnostic",
        xlabel="Number of Clusters (K)",
        ylabel="Within-Cluster Sum of Squares",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_validation_indices(results):
    """
    Plot internal validation indices against K.

    Input:
        results: Validation results for each value of K.

    Output:
        Displays separate validation index figures.
    """
    k_values = np.array([
        result["k"] for result in results
    ])

    silhouette = np.array([
        result["mean_silhouette"]
        for result in results
    ])

    calinski_harabasz = np.array([
        result["calinski_harabasz"]
        for result in results
    ])

    davies_bouldin = np.array([
        result["davies_bouldin"]
        for result in results
    ])

    figures = (
        (
            silhouette,
            "Average Silhouette Coefficient",
            "Average Silhouette",
        ),
        (
            calinski_harabasz,
            "Calinski-Harabasz Index",
            "Calinski-Harabasz",
        ),
        (
            davies_bouldin,
            "Davies-Bouldin Index",
            "Davies-Bouldin",
        ),
    )

    for values, title, ylabel in figures:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            k_values,
            values,
            marker="o",
            linewidth=1.5
        )

        ax.set(
            title=title,
            xlabel="Number of Clusters (K)",
            ylabel=ylabel,
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.show()


def plot_silhouette(result):
    """
    Plot observation-level Silhouette coefficients.

    Input:
        result: Validation result for one value of K.

    Output:
        Displays the Silhouette plot.
    """
    labels = result["labels"]
    silhouette = result["silhouette"]
    k = result["k"]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for cluster in range(k):
        values = np.sort(
            silhouette[labels == cluster]
        )

        size = len(values)
        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            alpha=0.6,
            label=f"Cluster {cluster + 1}"
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * size,
            str(cluster + 1)
        )

        y_lower = y_upper + 10

    ax.axvline(
        result["mean_silhouette"],
        linestyle="--",
        linewidth=1.2,
        label="Average Silhouette"
    )

    ax.axvline(
        0,
        linewidth=0.8
    )

    ax.set(
        title=f"Silhouette Plot — K = {k}",
        xlabel="Silhouette Coefficient",
        ylabel="Observations by Cluster",
    )
    ax.set_yticks([])
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_clustering(x, result):
    """
    Plot a K-Means solution for a selected value of K.

    Input:
        x: Observation matrix.
        result: Validation result for one value of K.

    Output:
        Displays the selected clustering.
    """
    labels = result["labels"]
    centroids = result["centroids"]
    k = result["k"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in range(k):
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
        title=f"K-Means Clustering — K = {k}",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Evaluate K-Means with internal validation measures.

    Input:
        None.

    Output:
        Prints validation results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n_per_cluster = 150
    n_init = 20
    k_values = np.arange(2, 9)
    selected_k = 4

    means = np.array([
        [-4.0, 2.5],
        [-1.5, -2.5],
        [2.0, -2.0],
        [4.0, 2.5],
    ])

    covariance = np.array([
        [0.8, 0.15],
        [0.15, 0.8],
    ])

    separator = "<>" * 36

    x = generate_data(
        n_per_cluster, means, covariance, rng
    )

    results = evaluate_clusterings(
        x, k_values, n_init, rng
    )

    selected_result = next(
        result
        for result in results
        if result["k"] == selected_k
    )

    # Main Results

    print(separator)
    print("K-Means Internal Validation")
    print(separator)
    print()

    print(f"Observations:          {len(x)}")
    print(f"Features:              {x.shape[1]}")
    print(f"K range:               {k_values[0]} to {k_values[-1]}")
    print(f"Initialisations per K: {n_init}")

    # Validation Summary

    print()
    print(separator)
    print("Validation Summary")
    print(separator)
    print()

    print(
        f"{'K':>3s} "
        f"{'WCSS':>14s} "
        f"{'Silhouette':>12s} "
        f"{'CH':>14s} "
        f"{'DB':>12s}"
    )

    for result in results:
        print(
            f"{result['k']:3d} "
            f"{result['objective']:14.4f} "
            f"{result['mean_silhouette']:12.4f} "
            f"{result['calinski_harabasz']:14.4f} "
            f"{result['davies_bouldin']:12.4f}"
        )

    # Diagnostic Extremes

    silhouette_values = np.array([
        result["mean_silhouette"]
        for result in results
    ])

    ch_values = np.array([
        result["calinski_harabasz"]
        for result in results
    ])

    db_values = np.array([
        result["davies_bouldin"]
        for result in results
    ])

    print()
    print(separator)
    print("Diagnostic Extremes")
    print(separator)
    print()

    print(
        f"Maximum Silhouette:    "
        f"K={k_values[np.argmax(silhouette_values)]}"
    )
    print(
        f"Maximum CH:            "
        f"K={k_values[np.argmax(ch_values)]}"
    )
    print(
        f"Minimum DB:            "
        f"K={k_values[np.argmin(db_values)]}"
    )

    print()
    print(
        "These values describe the extrema of each diagnostic; "
        "they do not define a uniquely correct K."
    )

    # Selected Clustering

    print()
    print(separator)
    print(f"Selected Clustering — K = {selected_k}")
    print(separator)
    print()

    print(
        f"WCSS:                  "
        f"{selected_result['objective']:.6f}"
    )
    print(
        f"Average Silhouette:    "
        f"{selected_result['mean_silhouette']:.6f}"
    )
    print(
        f"Calinski-Harabasz:     "
        f"{selected_result['calinski_harabasz']:.6f}"
    )
    print(
        f"Davies-Bouldin:        "
        f"{selected_result['davies_bouldin']:.6f}"
    )

    # Visualisation

    plot_elbow(results)
    plot_validation_indices(results)
    plot_clustering(x, selected_result)
    plot_silhouette(selected_result)


if __name__ == "__main__":
    main()