"""
Hierarchical Clustering — Computational Implementation.

NumPy: explicit distances, linkages and agglomerative algorithm.
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
# Pairwise Dissimilarities
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two observations.

    Input:
        x: First observation.
        y: Second observation.

    Output:
        Euclidean distance.
    """
    return np.sqrt(np.sum((x - y)**2))


def pairwise_distances(x):
    """
    Compute the symmetric pairwise Euclidean distance matrix.

    Input:
        x: Observation matrix.

    Output:
        distances: Pairwise distance matrix.
    """
    n = len(x)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(x[i], x[j])
            distances[i, j] = distance
            distances[j, i] = distance

    return distances


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Linkage Criteria
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def single_linkage(cluster_a, cluster_b, distances):
    """
    Compute single-linkage dissimilarity between two clusters.

    Input:
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.
        distances: Pairwise observation distance matrix.

    Output:
        Minimum cross-cluster dissimilarity.
    """
    cross_distances = distances[np.ix_(cluster_a, cluster_b)]
    return np.min(cross_distances)


def complete_linkage(cluster_a, cluster_b, distances):
    """
    Compute complete-linkage dissimilarity between two clusters.

    Input:
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.
        distances: Pairwise observation distance matrix.

    Output:
        Maximum cross-cluster dissimilarity.
    """
    cross_distances = distances[np.ix_(cluster_a, cluster_b)]
    return np.max(cross_distances)


def average_linkage(cluster_a, cluster_b, distances):
    """
    Compute average-linkage dissimilarity between two clusters.

    Input:
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.
        distances: Pairwise observation distance matrix.

    Output:
        Mean cross-cluster dissimilarity.
    """
    cross_distances = distances[np.ix_(cluster_a, cluster_b)]
    return np.mean(cross_distances)


def centroid_linkage(cluster_a, cluster_b, x):
    """
    Compute Euclidean distance between two cluster centroids.

    Input:
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.
        x: Observation matrix.

    Output:
        Euclidean distance between cluster centroids.
    """
    centroid_a = np.mean(x[cluster_a], axis=0)
    centroid_b = np.mean(x[cluster_b], axis=0)

    return euclidean_distance(centroid_a, centroid_b)


def cluster_distance(cluster_a, cluster_b, x, distances, linkage):
    """
    Compute inter-cluster dissimilarity for a linkage criterion.

    Input:
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.
        x: Observation matrix.
        distances: Pairwise observation distance matrix.
        linkage: Linkage criterion.

    Output:
        Inter-cluster dissimilarity.
    """
    if linkage == "single":
        return single_linkage(
            cluster_a, cluster_b, distances
        )

    if linkage == "complete":
        return complete_linkage(
            cluster_a, cluster_b, distances
        )

    if linkage == "average":
        return average_linkage(
            cluster_a, cluster_b, distances
        )

    if linkage == "centroid":
        return centroid_linkage(
            cluster_a, cluster_b, x
        )

    raise ValueError(f"Unknown linkage criterion: {linkage}")


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Agglomerative Hierarchical Clustering
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def find_closest_clusters(clusters, x, distances, linkage):
    """
    Find the pair of active clusters with minimum dissimilarity.

    Input:
        clusters: Dictionary of active clusters.
        x: Observation matrix.
        distances: Pairwise observation distance matrix.
        linkage: Linkage criterion.

    Output:
        cluster_a: Identifier of the first cluster.
        cluster_b: Identifier of the second cluster.
        minimum_distance: Dissimilarity between the clusters.
    """
    cluster_ids = list(clusters)
    cluster_a = None
    cluster_b = None
    minimum_distance = np.inf

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id_a = cluster_ids[i]
            id_b = cluster_ids[j]

            distance = cluster_distance(
                clusters[id_a],
                clusters[id_b],
                x,
                distances,
                linkage,
            )

            if distance < minimum_distance:
                cluster_a = id_a
                cluster_b = id_b
                minimum_distance = distance

    return cluster_a, cluster_b, minimum_distance


def agglomerative_clustering(x, linkage):
    """
    Construct an agglomerative hierarchical clustering tree.

    Input:
        x: Observation matrix.
        linkage: Linkage criterion.

    Output:
        linkage_matrix: Matrix describing the sequence of merges.
        merge_history: Detailed information for each merge.
    """
    n = len(x)
    distances = pairwise_distances(x)

    clusters = {
        i: [i]
        for i in range(n)
    }

    linkage_matrix = np.zeros((n - 1, 4))
    merge_history = []

    next_cluster_id = n

    for step in range(n - 1):
        cluster_a, cluster_b, distance = find_closest_clusters(
            clusters,
            x,
            distances,
            linkage,
        )

        members_a = clusters[cluster_a]
        members_b = clusters[cluster_b]
        merged_members = members_a + members_b
        merged_size = len(merged_members)

        linkage_matrix[step] = [
            cluster_a,
            cluster_b,
            distance,
            merged_size,
        ]

        merge_history.append({
            "step": step + 1,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "distance": distance,
            "size": merged_size,
        })

        del clusters[cluster_a]
        del clusters[cluster_b]

        clusters[next_cluster_id] = merged_members
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


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Hierarchy Diagnostics
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def cophenetic_distances(linkage_matrix, n_observations):
    """
    Compute cophenetic dissimilarities from a fitted hierarchy.

    Input:
        linkage_matrix: Matrix describing the sequence of merges.
        n_observations: Number of original observations.

    Output:
        cophenetic: Matrix of cophenetic dissimilarities.
    """
    clusters = {
        i: [i]
        for i in range(n_observations)
    }

    cophenetic = np.zeros(
        (n_observations, n_observations)
    )

    for step, merge in enumerate(linkage_matrix):
        cluster_a = int(merge[0])
        cluster_b = int(merge[1])
        height = merge[2]

        members_a = clusters[cluster_a]
        members_b = clusters[cluster_b]

        for i in members_a:
            for j in members_b:
                cophenetic[i, j] = height
                cophenetic[j, i] = height

        new_cluster_id = n_observations + step
        clusters[new_cluster_id] = members_a + members_b

    return cophenetic


def cophenetic_correlation(distances, cophenetic):
    """
    Compute correlation between original and cophenetic distances.

    Input:
        distances: Original pairwise distance matrix.
        cophenetic: Cophenetic distance matrix.

    Output:
        Correlation between distinct pairwise dissimilarities.
    """
    upper = np.triu_indices_from(distances, k=1)

    original_values = distances[upper]
    cophenetic_values = cophenetic[upper]

    return np.corrcoef(
        original_values,
        cophenetic_values,
    )[0, 1]


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


def plot_clusters(x, labels, linkage):
    """
    Plot clusters obtained by cutting the hierarchy.

    Input:
        x: Observation matrix.
        labels: Estimated cluster labels.
        linkage: Linkage criterion.

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
        title=f"Hierarchical Clustering — {linkage.title()} Linkage",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix, linkage):
    """
    Plot the dendrogram from the computed linkage matrix.

    Input:
        linkage_matrix: Matrix describing the sequence of merges.
        linkage: Linkage criterion.

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
        title=f"Dendrogram — {linkage.title()} Linkage",
        xlabel="Observations",
        ylabel="Dissimilarity",
    )

    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Generate data and apply agglomerative hierarchical clustering.

    Input:
        None.

    Output:
        Prints clustering results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n_per_cluster = 25
    n_clusters = 3
    linkage = "average"

    separator = "<>" * 36

    x, true_labels = generate_data(
        n_per_cluster,
        rng,
    )

    distances = pairwise_distances(x)

    linkage_matrix, merge_history = agglomerative_clustering(
        x,
        linkage,
    )

    labels = cut_tree(
        linkage_matrix,
        len(x),
        n_clusters,
    )

    cophenetic = cophenetic_distances(
        linkage_matrix,
        len(x),
    )

    correlation = cophenetic_correlation(
        distances,
        cophenetic,
    )

    heights = linkage_matrix[:, 2]

    # Main Results

    print(separator)
    print("Hierarchical Clustering")
    print(separator)
    print()

    print(f"Observations:          {len(x)}")
    print(f"Features:              {x.shape[1]}")
    print(f"True groups:           {len(np.unique(true_labels))}")
    print(f"Selected clusters:     {n_clusters}")
    print(f"Linkage criterion:     {linkage.title()}")

    # Agglomerative Clustering

    print()
    print(separator)
    print(f"Agglomerative Clustering — {linkage.title()} Linkage")
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
        f"First merge distance:  "
        f"{merge_history[0]['distance']:.6f}"
    )
    print(
        f"Last merge distance:   "
        f"{merge_history[-1]['distance']:.6f}"
    )

    # Hierarchy Diagnostics

    print()
    print(separator)
    print("Hierarchy Diagnostics")
    print(separator)
    print()

    print(
        f"Monotone heights:      "
        f"{np.all(np.diff(heights) >= -1e-12)}"
    )
    print(
        f"Cophenetic correlation:"
        f"  {correlation:.6f}"
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

    # Visualisation

    plot_data(
        x,
        true_labels,
    )

    plot_clusters(
        x,
        labels,
        linkage,
    )

    plot_dendrogram(
        linkage_matrix,
        linkage,
    )


if __name__ == "__main__":
    main()