"""
Ward's Minimum Variance Method — Computational Implementation.

NumPy: explicit WCSS, Ward criterion and agglomerative algorithm.
SciPy: dendrogram visualisation only.
Matplotlib: figures displayed, not saved.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import Libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Synthetic Data
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def generate_data(n_per_cluster, rng):
    """
    Generate synthetic observations from three Gaussian clusters.

    Input:
        n_per_cluster: Number of observations in each cluster.
        rng: NumPy random number generator.

    Output:
        x: Synthetic observations.
        true_labels: True cluster labels.
    """
    cluster_1 = rng.normal(
        loc=[-3.0, 0.0],
        scale=[0.7, 0.9],
        size=(n_per_cluster, 2),
    )

    cluster_2 = rng.normal(
        loc=[2.5, 3.0],
        scale=[0.8, 0.6],
        size=(n_per_cluster, 2),
    )

    cluster_3 = rng.normal(
        loc=[3.5, -2.5],
        scale=[0.9, 0.7],
        size=(n_per_cluster, 2),
    )

    x = np.vstack((cluster_1, cluster_2, cluster_3))

    true_labels = np.repeat(
        np.arange(3),
        n_per_cluster,
    )

    return x, true_labels


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Within-Cluster Sum of Squares
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def cluster_centroid(x, cluster):
    """
    Compute the centroid of a cluster.

    Input:
        x: Observation matrix.
        cluster: Observation indices in the cluster.

    Output:
        Cluster centroid.
    """
    return np.mean(x[cluster], axis=0)


def within_cluster_sum_squares(x, cluster):
    """
    Compute within-cluster sum of squares.

    Input:
        x: Observation matrix.
        cluster: Observation indices in the cluster.

    Output:
        Within-cluster sum of squares.
    """
    centroid = cluster_centroid(x, cluster)
    deviations = x[cluster] - centroid

    return np.sum(deviations**2)


def total_within_cluster_sum_squares(x, clusters):
    """
    Compute total within-cluster sum of squares for a partition.

    Input:
        x: Observation matrix.
        clusters: Collection of active clusters.

    Output:
        Total within-cluster sum of squares.
    """
    return sum(
        within_cluster_sum_squares(x, cluster)
        for cluster in clusters.values()
    )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Ward Criterion
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def ward_criterion(x, cluster_a, cluster_b):
    """
    Compute Ward's increase in within-cluster sum of squares.

    Input:
        x: Observation matrix.
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.

    Output:
        Increase in within-cluster sum of squares.
    """
    n_a = len(cluster_a)
    n_b = len(cluster_b)

    centroid_a = cluster_centroid(x, cluster_a)
    centroid_b = cluster_centroid(x, cluster_b)

    squared_distance = np.sum(
        (centroid_a - centroid_b)**2
    )

    return (
        n_a * n_b
        / (n_a + n_b)
        * squared_distance
    )


def direct_wcss_increase(x, cluster_a, cluster_b):
    """
    Compute the WCSS increase directly before and after a merge.

    Input:
        x: Observation matrix.
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.

    Output:
        Direct increase in within-cluster sum of squares.
    """
    merged_cluster = cluster_a + cluster_b

    wcss_before = (
        within_cluster_sum_squares(x, cluster_a)
        + within_cluster_sum_squares(x, cluster_b)
    )

    wcss_after = within_cluster_sum_squares(
        x,
        merged_cluster,
    )

    return wcss_after - wcss_before


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Agglomerative Ward Clustering
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def find_minimum_ward_merge(x, clusters):
    """
    Find the pair of clusters minimising Ward's criterion.

    Input:
        x: Observation matrix.
        clusters: Dictionary of active clusters.

    Output:
        cluster_a: Identifier of the first cluster.
        cluster_b: Identifier of the second cluster.
        minimum_increase: Minimum increase in WCSS.
    """
    cluster_ids = list(clusters)

    cluster_a = None
    cluster_b = None
    minimum_increase = np.inf

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id_a = cluster_ids[i]
            id_b = cluster_ids[j]

            increase = ward_criterion(
                x,
                clusters[id_a],
                clusters[id_b],
            )

            if increase < minimum_increase:
                cluster_a = id_a
                cluster_b = id_b
                minimum_increase = increase

    return cluster_a, cluster_b, minimum_increase


def ward_clustering(x):
    """
    Construct a hierarchy using Ward's minimum variance method.

    Input:
        x: Observation matrix.

    Output:
        linkage_matrix: Matrix describing the sequence of merges.
        merge_history: Detailed information for each merge.
    """
    n = len(x)

    clusters = {
        i: [i]
        for i in range(n)
    }

    linkage_matrix = np.zeros((n - 1, 4))
    merge_history = []

    next_cluster_id = n
    previous_wcss = 0.0

    for step in range(n - 1):
        cluster_a, cluster_b, increase = (
            find_minimum_ward_merge(
                x,
                clusters,
            )
        )

        members_a = clusters[cluster_a]
        members_b = clusters[cluster_b]
        merged_members = members_a + members_b

        direct_increase = direct_wcss_increase(
            x,
            members_a,
            members_b,
        )

        del clusters[cluster_a]
        del clusters[cluster_b]

        clusters[next_cluster_id] = merged_members

        total_wcss = total_within_cluster_sum_squares(
            x,
            clusters,
        )

        linkage_matrix[step] = [
            cluster_a,
            cluster_b,
            np.sqrt(2 * increase),
            len(merged_members),
        ]

        merge_history.append({
            "step": step + 1,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "increase": increase,
            "direct_increase": direct_increase,
            "total_wcss": total_wcss,
            "wcss_change": total_wcss - previous_wcss,
            "size": len(merged_members),
        })

        previous_wcss = total_wcss
        next_cluster_id += 1

    return linkage_matrix, merge_history


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Tree Cutting
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def cut_tree(linkage_matrix, n_observations, n_clusters):
    """
    Cut an agglomerative hierarchy into a fixed number of clusters.

    Input:
        linkage_matrix: Matrix describing the sequence of merges.
        n_observations: Number of original observations.
        n_clusters: Desired number of clusters.

    Output:
        labels: Cluster label for each observation.
    """
    if not 1 <= n_clusters <= n_observations:
        raise ValueError(
            "n_clusters must be between 1 and the number "
            "of observations."
        )

    clusters = {
        i: [i]
        for i in range(n_observations)
    }

    merges_to_apply = n_observations - n_clusters

    for step in range(merges_to_apply):
        cluster_a = int(linkage_matrix[step, 0])
        cluster_b = int(linkage_matrix[step, 1])
        new_cluster_id = n_observations + step

        merged_members = (
            clusters.pop(cluster_a)
            + clusters.pop(cluster_b)
        )

        clusters[new_cluster_id] = merged_members

    labels = np.empty(n_observations, dtype=int)

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda members: min(members),
    )

    for label, members in enumerate(ordered_clusters):
        labels[members] = label

    return labels


def partition_wcss(x, labels):
    """
    Compute WCSS for a partition represented by cluster labels.

    Input:
        x: Observation matrix.
        labels: Cluster label for each observation.

    Output:
        Total within-cluster sum of squares.
    """
    total = 0.0

    for label in np.unique(labels):
        cluster = np.flatnonzero(labels == label).tolist()
        total += within_cluster_sum_squares(
            x,
            cluster,
        )

    return total


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_data(x, true_labels):
    """
    Plot the synthetic observations and their generating groups.

    Input:
        x: Observation matrix.
        true_labels: True cluster labels.

    Output:
        Displays the synthetic observations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for label in np.unique(true_labels):
        mask = true_labels == label

        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=30,
            alpha=0.7,
            label=f"Group {label + 1}",
        )

    ax.set(
        title="Synthetic Data",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_clusters(x, labels):
    """
    Plot clusters obtained from Ward's hierarchy.

    Input:
        x: Observation matrix.
        labels: Estimated cluster labels.

    Output:
        Displays the estimated clusters.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for label in np.unique(labels):
        mask = labels == label

        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=30,
            alpha=0.7,
            label=f"Cluster {label + 1}",
        )

    ax.set(
        title="Ward's Minimum Variance Clustering",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_wcss_history(merge_history):
    """
    Plot total WCSS over the agglomerative sequence.

    Input:
        merge_history: Detailed information for each merge.

    Output:
        Displays total WCSS against the number of clusters.
    """
    n_observations = len(merge_history) + 1

    n_clusters = np.arange(
        n_observations,
        0,
        -1,
    )

    total_wcss = np.concatenate((
        [0.0],
        [
            merge["total_wcss"]
            for merge in merge_history
        ],
    ))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        n_clusters,
        total_wcss,
        marker="o",
        markersize=3,
        linewidth=1.2,
    )

    ax.set(
        title="Within-Cluster Sum of Squares",
        xlabel="Number of clusters",
        ylabel="Total WCSS",
    )

    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix):
    """
    Plot the dendrogram from the computed Ward hierarchy.

    Input:
        linkage_matrix: Matrix describing the sequence of merges.

    Output:
        Displays the dendrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    dendrogram(
        linkage_matrix,
        ax=ax,
        no_labels=True,
    )

    ax.set(
        title="Dendrogram — Ward's Method",
        xlabel="Observations",
        ylabel="Ward Distance",
    )

    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Generate data and apply Ward's minimum variance method.

    Input:
        None.

    Output:
        Prints clustering results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n_per_cluster = 25
    n_clusters = 3

    separator = "<>" * 36

    x, true_labels = generate_data(
        n_per_cluster,
        rng,
    )

    linkage_matrix, merge_history = ward_clustering(x)

    labels = cut_tree(
        linkage_matrix,
        len(x),
        n_clusters,
    )

    selected_wcss = partition_wcss(
        x,
        labels,
    )

    increases = np.array([
        merge["increase"]
        for merge in merge_history
    ])

    direct_increases = np.array([
        merge["direct_increase"]
        for merge in merge_history
    ])

    wcss_changes = np.array([
        merge["wcss_change"]
        for merge in merge_history
    ])

    # Main Results

    print(separator)
    print("Ward's Minimum Variance Method")
    print(separator)
    print()

    print(f"Observations:          {len(x)}")
    print(f"Features:              {x.shape[1]}")
    print(f"True groups:           {len(np.unique(true_labels))}")
    print(f"Selected clusters:     {n_clusters}")

    # Agglomerative Clustering

    print()
    print(separator)
    print("Agglomerative Ward Clustering")
    print(separator)
    print()

    print(f"Merges:                {len(merge_history)}")
    print(f"Initial clusters:      {len(x)}")
    print("Final clusters:        1")
    print(
        f"First merge:           "
        f"{merge_history[0]['cluster_a']} + "
        f"{merge_history[0]['cluster_b']}"
    )
    print(
        f"First WCSS increase:   "
        f"{merge_history[0]['increase']:.6f}"
    )
    print(
        f"Last WCSS increase:    "
        f"{merge_history[-1]['increase']:.6f}"
    )

    # Ward Identity

    print()
    print(separator)
    print("Ward Identity")
    print(separator)
    print()

    print(
        "Formula = direct:      "
        f"{np.allclose(increases, direct_increases)}"
    )
    print(
        "Formula = WCSS change: "
        f"{np.allclose(increases, wcss_changes)}"
    )
    print(
        f"Maximum difference:    "
        f"{np.max(np.abs(increases - direct_increases)):.3e}"
    )

    # Selected Partition

    print()
    print(separator)
    print(f"Partition — {n_clusters} Clusters")
    print(separator)
    print()

    for label in np.unique(labels):
        size = np.sum(labels == label)

        print(
            f"Cluster {label + 1}:             "
            f"{size} observations"
        )

    print()
    print(f"Partition WCSS:        {selected_wcss:.6f}")

    # Hierarchy Diagnostics

    print()
    print(separator)
    print("Hierarchy Diagnostics")
    print(separator)
    print()

    print(
        "Nondecreasing Ward criterion: "
        f"{np.all(np.diff(increases) >= -1e-12)}"
    )
    print(
        f"Final total WCSS:      "
        f"{merge_history[-1]['total_wcss']:.6f}"
    )
    print(
        f"Total data variance:   "
        f"{within_cluster_sum_squares(
            x, list(range(len(x)))
        ):.6f}"
    )

    # Visualisation

    plot_data(
        x,
        true_labels,
    )

    plot_clusters(
        x,
        labels,
    )

    plot_wcss_history(
        merge_history,
    )

    plot_dendrogram(
        linkage_matrix,
    )


if __name__ == "__main__":
    main()