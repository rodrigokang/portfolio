"""
K-Means Initialisation — Computational Implementation.

NumPy: explicit Lloyd's algorithm and repeated initialisations.
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
        objective_history: Objective values during optimisation.
        converged: Whether the convergence criterion was reached.
        iterations: Number of centroid updates.
    """
    centroids = np.asarray(
        initial_centroids, dtype=float
    ).copy()

    objective_history = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        labels = assign_clusters(x, centroids)
        objective_history.append(
            objective_function(x, labels, centroids)
        )

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
    final_objective = objective_function(
        x, labels, centroids
    )

    if not np.isclose(objective_history[-1], final_objective):
        objective_history.append(final_objective)

    return (
        labels,
        centroids,
        np.asarray(objective_history),
        converged,
        iteration,
    )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Multiple Initialisations
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

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


def multiple_initialisations(x, k, n_init, rng):
    """
    Fit K-Means from several random initial configurations.

    Input:
        x: Observation matrix.
        k: Number of clusters.
        n_init: Number of independent initialisations.
        rng: NumPy random number generator.

    Output:
        runs: Results from all valid K-Means runs.
        best_run: Run with the smallest final objective.
    """
    runs = []

    for run in range(n_init):
        initial_centroids = random_centroids(x, k, rng)

        (
            labels,
            centroids,
            objective_history,
            converged,
            iterations,
        ) = k_means(x, initial_centroids)

        if not converged:
            continue

        runs.append({
            "run": run + 1,
            "labels": labels,
            "centroids": centroids,
            "objective_history": objective_history,
            "final_objective": objective_history[-1],
            "iterations": iterations,
        })

    if not runs:
        raise RuntimeError(
            "No initialisation produced a valid solution."
        )

    best_run = min(
        runs, key=lambda result: result["final_objective"]
    )

    return runs, best_run


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Local Minimum Example
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def local_minimum_example():
    """
    Compare two stable K-Means solutions for {0, 6, 10}.

    Input:
        None.

    Output:
        poor_result: Stable solution with objective 18.
        best_result: Stable solution with objective 8.
    """
    x = np.array([
        [0.0],
        [6.0],
        [10.0],
    ])

    poor_initial = np.array([
        [3.0],
        [10.0],
    ])

    best_initial = np.array([
        [0.0],
        [8.0],
    ])

    poor_result = k_means(x, poor_initial)
    best_result = k_means(x, best_initial)

    return poor_result, best_result


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_local_minima(poor_result, best_result):
    """
    Plot two stable solutions for the one-dimensional example.

    Input:
        poor_result: K-Means solution with the larger objective.
        best_result: K-Means solution with the smaller objective.

    Output:
        Displays the two stable K-Means solutions.
    """
    x = np.array([0.0, 6.0, 10.0])

    solutions = (
        ("Local Minimum", poor_result),
        ("Better Minimum", best_result),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, (title, result) in zip(axes, solutions):
        labels, centroids, history, _, _ = result

        for cluster in range(len(centroids)):
            members = x[labels == cluster]

            ax.scatter(
                members,
                np.zeros(len(members)),
                s=70,
                label=f"Cluster {cluster + 1}"
            )

            ax.scatter(
                centroids[cluster, 0],
                0,
                marker="X",
                s=150,
                edgecolor="black",
                linewidth=1
            )

        ax.set(
            title=f"{title}: J = {history[-1]:.0f}",
            xlabel="x",
            yticks=[],
        )
        ax.set_xlim(-1, 11)
        ax.grid(axis="x", alpha=0.2)

    axes[0].legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_objective_paths(runs):
    """
    Plot objective trajectories for repeated initialisations.

    Input:
        runs: Results from multiple K-Means initialisations.

    Output:
        Displays the objective trajectory for every run.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for result in runs:
        history = result["objective_history"]
        iterations = np.arange(len(history))

        ax.plot(
            iterations,
            history,
            marker="o",
            markersize=3,
            linewidth=1,
            alpha=0.65
        )

    ax.set(
        title="Objective Paths Across Initialisations",
        xlabel="Iteration",
        ylabel="Within-Cluster Sum of Squares",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_final_objectives(runs, best_run):
    """
    Plot final objective values across initialisations.

    Input:
        runs: Results from multiple K-Means initialisations.
        best_run: Run with the smallest final objective.

    Output:
        Displays final objective values for all runs.
    """
    run_numbers = np.array([
        result["run"] for result in runs
    ])

    objectives = np.array([
        result["final_objective"] for result in runs
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        run_numbers,
        objectives,
        s=45,
        label="Final objective"
    )

    ax.axhline(
        best_run["final_objective"],
        linestyle="--",
        linewidth=1,
        label="Best objective"
    )

    ax.set(
        title="Final Objective Across Initialisations",
        xlabel="Initialisation",
        ylabel="Within-Cluster Sum of Squares",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_best_clustering(x, best_run):
    """
    Plot the best solution from repeated initialisations.

    Input:
        x: Observation matrix.
        best_run: Run with the smallest final objective.

    Output:
        Displays the retained K-Means solution.
    """
    labels = best_run["labels"]
    centroids = best_run["centroids"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in range(len(centroids)):
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
        title="Best K-Means Solution",
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
    Study local minima and repeated K-Means initialisations.

    Input:
        None.

    Output:
        Prints optimisation results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    separator = "<>" * 36

    # Local Minimum Example

    poor_result, best_result = local_minimum_example()

    poor_objective = poor_result[2][-1]
    best_objective = best_result[2][-1]

    print(separator)
    print("K-Means Initialisation")
    print(separator)
    print()

    print("One-Dimensional Local Minimum Example")
    print()

    print(f"Observations:          [0, 6, 10]")
    print(f"Clusters:              2")
    print(f"Local minimum:         {poor_objective:.6f}")
    print(f"Better minimum:        {best_objective:.6f}")
    print(
        f"Same objective:        "
        f"{np.isclose(poor_objective, best_objective)}"
    )

    # Synthetic Experiment

    n_per_cluster = 150
    k = 4
    n_init = 20

    means = np.array([
        [-3.0, 2.0],
        [-1.0, -2.0],
        [2.0, -1.5],
        [3.0, 2.5],
    ])

    covariance = np.array([
        [1.2, 0.25],
        [0.25, 1.0],
    ])

    x = generate_data(
        n_per_cluster, means, covariance, rng
    )

    runs, best_run = multiple_initialisations(
        x, k, n_init, rng
    )

    objectives = np.array([
        result["final_objective"] for result in runs
    ])

    monotonic_runs = [
        np.all(
            np.diff(result["objective_history"]) <= 1e-10
        )
        for result in runs
    ]

    # Multiple Initialisations

    print()
    print(separator)
    print("Multiple Initialisations")
    print(separator)
    print()

    print(f"Observations:          {len(x)}")
    print(f"Features:              {x.shape[1]}")
    print(f"Clusters:              {k}")
    print(f"Requested runs:        {n_init}")
    print(f"Converged runs:        {len(runs)}")
    print(f"Best run:              {best_run['run']}")
    print(f"Best objective:        {objectives.min():.6f}")
    print(f"Worst objective:       {objectives.max():.6f}")
    print(f"Mean objective:        {objectives.mean():.6f}")
    print(f"Std. objective:        {objectives.std():.6f}")
    print(
        f"All paths monotonic:   "
        f"{all(monotonic_runs)}"
    )

    # Run Summary

    print()
    print(separator)
    print("Run Summary")
    print(separator)
    print()

    print(
        f"{'Run':>4s} "
        f"{'Iterations':>12s} "
        f"{'Final Objective':>18s}"
    )

    for result in runs:
        marker = "*" if result["run"] == best_run["run"] else " "

        print(
            f"{result['run']:4d} "
            f"{result['iterations']:12d} "
            f"{result['final_objective']:18.6f} "
            f"{marker}"
        )

    print()
    print("* Best retained solution")

    # Visualisation

    plot_local_minima(poor_result, best_result)
    plot_objective_paths(runs)
    plot_final_objectives(runs, best_run)
    plot_best_clustering(x, best_run)


if __name__ == "__main__":
    main()