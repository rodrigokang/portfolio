"""
K-Means Clustering — Computational Implementation.

NumPy: synthetic data and explicit Lloyd's algorithm.
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
        true_labels: Generating cluster labels.
    """
    observations = []
    labels = []

    for k, mean in enumerate(means):
        cluster = rng.multivariate_normal(
            mean, covariance, size=n_per_cluster
        )
        observations.append(cluster)
        labels.append(np.full(n_per_cluster, k))

    x = np.vstack(observations)
    true_labels = np.concatenate(labels)

    order = rng.permutation(len(x))
    return x[order], true_labels[order]


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
        distances: Squared distance from each observation to each
            centroid.
    """
    differences = x[:, None, :] - centroids[None, :, :]
    return np.sum(differences**2, axis=2)


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
# Initialisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def initialise_centroids(x, k, rng):
    """
    Select distinct observations as initial centroids.

    Input:
        x: Observation matrix.
        k: Number of clusters.
        rng: NumPy random number generator.

    Output:
        Initial cluster centroids.
    """
    if k < 1 or k > len(x):
        raise ValueError(
            "Number of clusters must be between 1 and N."
        )

    indices = rng.choice(len(x), size=k, replace=False)
    return x[indices].copy()


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


def update_centroids(x, labels, centroids, rng):
    """
    Update centroids using the observations in each cluster.

    Empty clusters are reinitialised using a randomly selected
    observation.

    Input:
        x: Observation matrix.
        labels: Cluster assignment for each observation.
        centroids: Current cluster centroid matrix.
        rng: NumPy random number generator.

    Output:
        new_centroids: Updated cluster centroids.
        empty_clusters: Indices of reinitialised empty clusters.
    """
    k = len(centroids)
    new_centroids = np.empty_like(centroids)
    empty_clusters = []

    for cluster in range(k):
        members = x[labels == cluster]

        if len(members) == 0:
            index = rng.integers(len(x))
            new_centroids[cluster] = x[index]
            empty_clusters.append(cluster)
        else:
            new_centroids[cluster] = np.mean(members, axis=0)

    return new_centroids, empty_clusters


def k_means(x, k, rng, tolerance=1e-6, max_iterations=100):
    """
    Fit K-Means using Lloyd's alternating optimisation algorithm.

    Input:
        x: Observation matrix.
        k: Number of clusters.
        rng: NumPy random number generator.
        tolerance: Maximum centroid movement for convergence.
        max_iterations: Maximum number of centroid updates.

    Output:
        labels: Final cluster assignments.
        centroids: Final cluster centroids.
        objective_history: Objective value after each assignment.
        centroid_history: Centroids from initialisation to solution.
        converged: Whether the convergence criterion was reached.
        iterations: Number of centroid updates.
        empty_count: Number of empty-cluster reinitialisations.
    """
    centroids = initialise_centroids(x, k, rng)
    centroid_history = [centroids.copy()]
    objective_history = []
    empty_count = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        labels = assign_clusters(x, centroids)
        objective_history.append(
            objective_function(x, labels, centroids)
        )

        new_centroids, empty_clusters = update_centroids(
            x, labels, centroids, rng
        )
        empty_count += len(empty_clusters)

        movement = np.max(
            np.linalg.norm(new_centroids - centroids, axis=1)
        )

        centroids = new_centroids
        centroid_history.append(centroids.copy())

        if movement <= tolerance and not empty_clusters:
            converged = True
            break

    labels = assign_clusters(x, centroids)
    final_objective = objective_function(x, labels, centroids)

    if not np.isclose(objective_history[-1], final_objective):
        objective_history.append(final_objective)

    return (
        labels,
        centroids,
        np.asarray(objective_history),
        np.asarray(centroid_history),
        converged,
        iteration,
        empty_count,
    )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_clustering(x, labels, centroids):
    """
    Plot observations and final cluster centroids.

    Input:
        x: Observation matrix.
        labels: Final cluster assignments.
        centroids: Final cluster centroids.

    Output:
        Displays the final clustering.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in range(len(centroids)):
        members = x[labels == cluster]
        ax.scatter(
            members[:, 0], members[:, 1],
            s=20, alpha=0.55,
            label=f"Cluster {cluster + 1}"
        )

    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        marker="X", s=150, edgecolor="black",
        linewidth=1, label="Centroids"
    )

    ax.set(
        title="K-Means Clustering",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_objective(objective_history):
    """
    Plot the K-Means objective over Lloyd iterations.

    Input:
        objective_history: Objective values during optimisation.

    Output:
        Displays the objective function trajectory.
    """
    iterations = np.arange(len(objective_history))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        iterations, objective_history,
        marker="o", linewidth=1.5
    )

    ax.set(
        title="K-Means Objective Function",
        xlabel="Iteration",
        ylabel="Within-Cluster Sum of Squares",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_centroid_paths(x, labels, centroid_history):
    """
    Plot centroid trajectories during Lloyd's algorithm.

    Input:
        x: Observation matrix.
        labels: Final cluster assignments.
        centroid_history: Centroids at each iteration.

    Output:
        Displays the centroid optimisation paths.
    """
    final_centroids = centroid_history[-1]
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in range(len(final_centroids)):
        members = x[labels == cluster]
        path = centroid_history[:, cluster, :]

        ax.scatter(
            members[:, 0], members[:, 1],
            s=18, alpha=0.3
        )
        ax.plot(
            path[:, 0], path[:, 1],
            marker="o", linewidth=1.5,
            label=f"Centroid {cluster + 1}"
        )
        ax.scatter(
            path[-1, 0], path[-1, 1],
            marker="X", s=130, edgecolor="black",
            linewidth=1
        )

    ax.set(
        title="Centroid Paths",
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
    Generate data and fit K-Means using Lloyd's algorithm.

    Input:
        None.

    Output:
        Prints clustering results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n_per_cluster = 200
    k = 3
    tolerance = 1e-6
    max_iterations = 100

    means = np.array([
        [-3.0, 2.0],
        [0.0, -3.0],
        [3.0, 2.0],
    ])

    covariance = np.array([
        [0.7, 0.15],
        [0.15, 0.7],
    ])

    separator = "<>" * 36

    x, _ = generate_data(
        n_per_cluster, means, covariance, rng
    )

    (
        labels,
        centroids,
        objective_history,
        centroid_history,
        converged,
        iterations,
        empty_count,
    ) = k_means(
        x, k, rng, tolerance, max_iterations
    )

    cluster_sizes = np.bincount(labels, minlength=k)
    monotonic = np.all(np.diff(objective_history) <= 1e-10)

    # Main Results

    print(separator)
    print("K-Means Clustering")
    print(separator)
    print()

    print(f"Observations:          {len(x)}")
    print(f"Features:              {x.shape[1]}")
    print(f"Clusters:              {k}")
    print(f"Iterations:            {iterations}")
    print(f"Converged:             {converged}")

    # Objective Function

    print()
    print(separator)
    print("Objective Function")
    print(separator)
    print()

    print(f"Initial objective:     {objective_history[0]:.6f}")
    print(f"Final objective:       {objective_history[-1]:.6f}")
    print(
        f"Objective reduction:   "
        f"{objective_history[0] - objective_history[-1]:.6f}"
    )
    print(f"Monotonic:             {monotonic}")

    # Final Centroids

    print()
    print(separator)
    print("Final Centroids")
    print(separator)
    print()

    for cluster, centroid in enumerate(centroids):
        print(
            f"Cluster {cluster + 1}:             "
            f"{np.array2string(centroid, precision=6)}"
        )

    # Cluster Sizes

    print()
    print(separator)
    print("Cluster Sizes")
    print(separator)
    print()

    for cluster, size in enumerate(cluster_sizes):
        print(f"Cluster {cluster + 1}:             {size}")

    # Numerical Details

    print()
    print(separator)
    print("Numerical Details")
    print(separator)
    print()

    print(f"Tolerance:             {tolerance:.1e}")
    print(f"Maximum iterations:    {max_iterations}")
    print(f"Empty reinitialisations: {empty_count}")

    # Visualisation

    plot_clustering(x, labels, centroids)
    plot_objective(objective_history)
    plot_centroid_paths(x, labels, centroid_history)


if __name__ == "__main__":
    main()