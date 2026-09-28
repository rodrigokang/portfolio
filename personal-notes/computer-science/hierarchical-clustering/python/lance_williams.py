"""
Lance-Williams Recurrence — Computational Implementation.

NumPy: explicit dissimilarities and Lance-Williams updates.
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
# Initial Dissimilarities
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


def initial_dissimilarities(x, linkage):
    """
    Compute dissimilarities between singleton clusters.

    Input:
        x: Observation matrix.
        linkage: Linkage criterion.

    Output:
        dissimilarities: Dictionary of pairwise dissimilarities.
    """
    n = len(x)
    dissimilarities = {}

    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(x[i], x[j])

            if linkage in ("centroid", "ward"):
                distance = distance**2

            dissimilarities[(i, j)] = distance

    return dissimilarities


def get_dissimilarity(dissimilarities, cluster_a, cluster_b):
    """
    Retrieve dissimilarity between two active clusters.

    Input:
        dissimilarities: Dictionary of pairwise dissimilarities.
        cluster_a: Identifier of the first cluster.
        cluster_b: Identifier of the second cluster.

    Output:
        Dissimilarity between the clusters.
    """
    key = tuple(sorted((cluster_a, cluster_b)))
    return dissimilarities[key]


def set_dissimilarity(dissimilarities, cluster_a, cluster_b,
                      value):
    """
    Store dissimilarity between two active clusters.

    Input:
        dissimilarities: Dictionary of pairwise dissimilarities.
        cluster_a: Identifier of the first cluster.
        cluster_b: Identifier of the second cluster.
        value: Dissimilarity to store.

    Output:
        None.
    """
    key = tuple(sorted((cluster_a, cluster_b)))
    dissimilarities[key] = value


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Lance-Williams Coefficients
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def lance_williams_coefficients(linkage, size_a, size_b, size_c):
    """
    Return Lance-Williams coefficients for a linkage criterion.

    Input:
        linkage: Linkage criterion.
        size_a: Number of observations in cluster A.
        size_b: Number of observations in cluster B.
        size_c: Number of observations in cluster C.

    Output:
        alpha_a: Coefficient for d(A, C).
        alpha_b: Coefficient for d(B, C).
        beta: Coefficient for d(A, B).
        gamma: Coefficient for the absolute difference term.
    """
    if linkage == "single":
        return 0.5, 0.5, 0.0, -0.5

    if linkage == "complete":
        return 0.5, 0.5, 0.0, 0.5

    if linkage == "average":
        total = size_a + size_b

        alpha_a = size_a / total
        alpha_b = size_b / total

        return alpha_a, alpha_b, 0.0, 0.0

    if linkage == "centroid":
        total = size_a + size_b

        alpha_a = size_a / total
        alpha_b = size_b / total
        beta = -(size_a * size_b) / total**2

        return alpha_a, alpha_b, beta, 0.0

    if linkage == "ward":
        total = size_a + size_b + size_c

        alpha_a = (size_a + size_c) / total
        alpha_b = (size_b + size_c) / total
        beta = -size_c / total

        return alpha_a, alpha_b, beta, 0.0

    raise ValueError(f"Unknown linkage criterion: {linkage}")


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Lance-Williams Update
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def lance_williams_update(d_ac, d_bc, d_ab, linkage,
                          size_a, size_b, size_c):
    """
    Update a dissimilarity using the Lance-Williams recurrence.

    Input:
        d_ac: Dissimilarity between clusters A and C.
        d_bc: Dissimilarity between clusters B and C.
        d_ab: Dissimilarity between clusters A and B.
        linkage: Linkage criterion.
        size_a: Number of observations in cluster A.
        size_b: Number of observations in cluster B.
        size_c: Number of observations in cluster C.

    Output:
        Updated dissimilarity between A union B and C.
    """
    alpha_a, alpha_b, beta, gamma = (
        lance_williams_coefficients(
            linkage,
            size_a,
            size_b,
            size_c,
        )
    )

    updated = (
        alpha_a * d_ac
        + alpha_b * d_bc
        + beta * d_ab
        + gamma * abs(d_ac - d_bc)
    )

    return updated


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Agglomerative Hierarchical Clustering
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def find_closest_clusters(clusters, dissimilarities):
    """
    Find the pair of active clusters with minimum dissimilarity.

    Input:
        clusters: Dictionary of active cluster sizes.
        dissimilarities: Dictionary of pairwise dissimilarities.

    Output:
        cluster_a: Identifier of the first cluster.
        cluster_b: Identifier of the second cluster.
        minimum_distance: Minimum active dissimilarity.
    """
    cluster_ids = list(clusters)

    cluster_a = None
    cluster_b = None
    minimum_distance = np.inf

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id_a = cluster_ids[i]
            id_b = cluster_ids[j]

            distance = get_dissimilarity(
                dissimilarities,
                id_a,
                id_b,
            )

            if distance < minimum_distance:
                cluster_a = id_a
                cluster_b = id_b
                minimum_distance = distance

    return cluster_a, cluster_b, minimum_distance


def dendrogram_height(dissimilarity, linkage):
    """
    Convert internal dissimilarity to dendrogram height.

    Input:
        dissimilarity: Internal merge dissimilarity.
        linkage: Linkage criterion.

    Output:
        Height used in the dendrogram.
    """
    dissimilarity = max(dissimilarity, 0.0)

    if linkage == "centroid":
        return np.sqrt(dissimilarity)

    if linkage == "ward":
        return np.sqrt(2 * dissimilarity)

    return dissimilarity


def lance_williams_clustering(x, linkage):
    """
    Construct a hierarchy using Lance-Williams updates.

    Input:
        x: Observation matrix.
        linkage: Linkage criterion.

    Output:
        linkage_matrix: Matrix describing the sequence of merges.
        merge_history: Detailed information for each merge.
    """
    n = len(x)

    clusters = {
        i: 1
        for i in range(n)
    }

    dissimilarities = initial_dissimilarities(
        x,
        linkage,
    )

    linkage_matrix = np.zeros((n - 1, 4))
    merge_history = []

    next_cluster_id = n

    for step in range(n - 1):
        cluster_a, cluster_b, distance = find_closest_clusters(
            clusters,
            dissimilarities,
        )

        size_a = clusters[cluster_a]
        size_b = clusters[cluster_b]
        merged_size = size_a + size_b

        other_clusters = [
            cluster_id
            for cluster_id in clusters
            if cluster_id not in (cluster_a, cluster_b)
        ]

        new_distances = {}

        for cluster_c in other_clusters:
            size_c = clusters[cluster_c]

            d_ac = get_dissimilarity(
                dissimilarities,
                cluster_a,
                cluster_c,
            )

            d_bc = get_dissimilarity(
                dissimilarities,
                cluster_b,
                cluster_c,
            )

            updated = lance_williams_update(
                d_ac,
                d_bc,
                distance,
                linkage,
                size_a,
                size_b,
                size_c,
            )

            new_distances[cluster_c] = updated

        height = dendrogram_height(
            distance,
            linkage,
        )

        linkage_matrix[step] = [
            cluster_a,
            cluster_b,
            height,
            merged_size,
        ]

        merge_history.append({
            "step": step + 1,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "dissimilarity": distance,
            "height": height,
            "size": merged_size,
        })

        del clusters[cluster_a]
        del clusters[cluster_b]

        clusters[next_cluster_id] = merged_size

        for cluster_c, updated in new_distances.items():
            set_dissimilarity(
                dissimilarities,
                next_cluster_id,
                cluster_c,
                updated,
            )

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
# Direct Linkage Verification
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def direct_cluster_dissimilarity(x, cluster_a, cluster_b, linkage):
    """
    Compute linkage dissimilarity directly from observations.

    Input:
        x: Observation matrix.
        cluster_a: Observation indices in the first cluster.
        cluster_b: Observation indices in the second cluster.
        linkage: Linkage criterion.

    Output:
        Direct inter-cluster dissimilarity.
    """
    if linkage in ("single", "complete", "average"):
        differences = (
            x[cluster_a][:, None, :]
            - x[cluster_b][None, :, :]
        )

        cross_distances = np.sqrt(
            np.sum(differences**2, axis=2)
        )

        if linkage == "single":
            return np.min(cross_distances)

        if linkage == "complete":
            return np.max(cross_distances)

        return np.mean(cross_distances)

    centroid_a = np.mean(x[cluster_a], axis=0)
    centroid_b = np.mean(x[cluster_b], axis=0)

    squared_distance = np.sum(
        (centroid_a - centroid_b)**2
    )

    if linkage == "centroid":
        return squared_distance

    if linkage == "ward":
        n_a = len(cluster_a)
        n_b = len(cluster_b)

        return (
            n_a * n_b
            / (n_a + n_b)
            * squared_distance
        )

    raise ValueError(f"Unknown linkage criterion: {linkage}")


def verify_merges(x, linkage_matrix, merge_history, linkage):
    """
    Compare recursive merge values with direct calculations.

    Input:
        x: Observation matrix.
        linkage_matrix: Matrix describing the sequence of merges.
        merge_history: Detailed information for each merge.
        linkage: Linkage criterion.

    Output:
        recursive_values: Lance-Williams merge dissimilarities.
        direct_values: Direct merge dissimilarities.
    """
    n = len(x)

    clusters = {
        i: [i]
        for i in range(n)
    }

    recursive_values = []
    direct_values = []

    for step, merge in enumerate(linkage_matrix):
        cluster_a = int(merge[0])
        cluster_b = int(merge[1])

        members_a = clusters[cluster_a]
        members_b = clusters[cluster_b]

        direct = direct_cluster_dissimilarity(
            x,
            members_a,
            members_b,
            linkage,
        )

        recursive_values.append(
            merge_history[step]["dissimilarity"]
        )

        direct_values.append(direct)

        clusters[n + step] = members_a + members_b

    return (
        np.asarray(recursive_values),
        np.asarray(direct_values),
    )


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
        title=f"Lance-Williams — {linkage.title()} Linkage",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix, linkage):
    """
    Plot the dendrogram from the computed hierarchy.

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
    Apply hierarchical clustering with Lance-Williams updates.

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

    linkage_matrix, merge_history = (
        lance_williams_clustering(
            x,
            linkage,
        )
    )

    labels = cut_tree(
        linkage_matrix,
        len(x),
        n_clusters,
    )

    recursive_values, direct_values = verify_merges(
        x,
        linkage_matrix,
        merge_history,
        linkage,
    )

    heights = linkage_matrix[:, 2]

    # Main Results

    print(separator)
    print("Lance-Williams Recurrence")
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
        f"First dissimilarity:   "
        f"{merge_history[0]['dissimilarity']:.6f}"
    )
    print(
        f"Last dissimilarity:    "
        f"{merge_history[-1]['dissimilarity']:.6f}"
    )

    # Lance-Williams Verification

    print()
    print(separator)
    print("Lance-Williams Verification")
    print(separator)
    print()

    print(
        "Recursive = direct:    "
        f"{np.allclose(recursive_values, direct_values)}"
    )
    print(
        f"Maximum difference:    "
        f"{np.max(np.abs(
            recursive_values - direct_values
        )):.3e}"
    )

    # Hierarchy Diagnostics

    print()
    print(separator)
    print("Hierarchy Diagnostics")
    print(separator)
    print()

    print(
        "Monotone heights:      "
        f"{np.all(np.diff(heights) >= -1e-12)}"
    )

    if linkage in ("centroid", "ward"):
        print(
            "Internal scale:       "
            "Squared dissimilarity"
        )
    else:
        print(
            "Internal scale:       "
            "Dissimilarity"
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