"""
Divisive Hierarchical Clustering — Computational Implementation.

NumPy: explicit dissimilarities and divisive clustering algorithm.
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


def average_dissimilarity(observation, group, distances):
    """
    Compute average dissimilarity from an observation to a group.

    Input:
        observation: Observation index.
        group: Observation indices defining the group.
        distances: Pairwise distance matrix.

    Output:
        Average dissimilarity to the group.
    """
    if len(group) == 0:
        raise ValueError("The comparison group cannot be empty.")

    return np.mean(
        distances[observation, group]
    )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Splinter Procedure
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def initial_splinter(cluster, distances):
    """
    Select the initial observation for the splinter group.

    Input:
        cluster: Observation indices in the cluster.
        distances: Pairwise distance matrix.

    Output:
        Observation with maximum average dissimilarity.
    """
    average_distances = []

    for observation in cluster:
        remaining = [
            member
            for member in cluster
            if member != observation
        ]

        average_distances.append(
            average_dissimilarity(
                observation,
                remaining,
                distances,
            )
        )

    position = np.argmax(average_distances)

    return cluster[position]


def transfer_score(observation, remainder, splinter, distances):
    """
    Compute the transfer score for a candidate observation.

    Input:
        observation: Candidate observation index.
        remainder: Current remainder group.
        splinter: Current splinter group.
        distances: Pairwise distance matrix.

    Output:
        Difference between average dissimilarities to both groups.
    """
    remaining_members = [
        member
        for member in remainder
        if member != observation
    ]

    if len(remaining_members) == 0:
        return -np.inf

    to_remainder = average_dissimilarity(
        observation,
        remaining_members,
        distances,
    )

    to_splinter = average_dissimilarity(
        observation,
        splinter,
        distances,
    )

    return to_remainder - to_splinter


def split_cluster(cluster, distances):
    """
    Split one cluster using the Macnaughton-Smith procedure.

    Input:
        cluster: Observation indices in the cluster.
        distances: Pairwise distance matrix.

    Output:
        remainder: Observations remaining in the original group.
        splinter: Observations transferred to the splinter group.
        transfer_history: Details of successive transfers.
    """
    if len(cluster) < 2:
        raise ValueError(
            "A cluster must contain at least two observations."
        )

    seed = initial_splinter(
        cluster,
        distances,
    )

    remainder = [
        observation
        for observation in cluster
        if observation != seed
    ]

    splinter = [seed]
    transfer_history = []

    while len(remainder) > 1:
        scores = {
            observation: transfer_score(
                observation,
                remainder,
                splinter,
                distances,
            )
            for observation in remainder
        }

        observation = max(
            scores,
            key=scores.get,
        )

        maximum_score = scores[observation]

        if maximum_score <= 0:
            break

        remainder.remove(observation)
        splinter.append(observation)

        transfer_history.append({
            "observation": observation,
            "score": maximum_score,
        })

    return remainder, splinter, transfer_history


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Cluster Selection
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def cluster_diameter(cluster, distances):
    """
    Compute the maximum dissimilarity within a cluster.

    Input:
        cluster: Observation indices in the cluster.
        distances: Pairwise distance matrix.

    Output:
        Maximum within-cluster dissimilarity.
    """
    if len(cluster) < 2:
        return 0.0

    submatrix = distances[np.ix_(cluster, cluster)]

    return np.max(submatrix)


def select_cluster_to_split(clusters, distances):
    """
    Select the active cluster with the largest diameter.

    Input:
        clusters: Dictionary of active clusters.
        distances: Pairwise distance matrix.

    Output:
        cluster_id: Identifier of the cluster to split.
        diameter: Diameter of the selected cluster.
    """
    cluster_id = None
    maximum_diameter = -np.inf

    for current_id, cluster in clusters.items():
        if len(cluster) < 2:
            continue

        diameter = cluster_diameter(
            cluster,
            distances,
        )

        if diameter > maximum_diameter:
            cluster_id = current_id
            maximum_diameter = diameter

    return cluster_id, maximum_diameter


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Divisive Hierarchical Clustering
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def divisive_clustering(x, n_clusters):
    """
    Construct a divisive hierarchy to a selected number of clusters.

    Input:
        x: Observation matrix.
        n_clusters: Desired number of clusters.

    Output:
        clusters: Final collection of clusters.
        split_history: Detailed information for each split.
    """
    n = len(x)

    if not 1 <= n_clusters <= n:
        raise ValueError(
            "n_clusters must be between 1 and the number "
            "of observations."
        )

    distances = pairwise_distances(x)

    clusters = {
        0: list(range(n))
    }

    split_history = []
    next_cluster_id = 1

    while len(clusters) < n_clusters:
        cluster_id, diameter = select_cluster_to_split(
            clusters,
            distances,
        )

        if cluster_id is None:
            break

        original_cluster = clusters.pop(cluster_id)

        remainder, splinter, transfers = split_cluster(
            original_cluster,
            distances,
        )

        remainder_id = next_cluster_id
        splinter_id = next_cluster_id + 1

        clusters[remainder_id] = remainder
        clusters[splinter_id] = splinter

        split_history.append({
            "step": len(split_history) + 1,
            "parent": cluster_id,
            "parent_size": len(original_cluster),
            "diameter": diameter,
            "remainder_id": remainder_id,
            "remainder_size": len(remainder),
            "splinter_id": splinter_id,
            "splinter_size": len(splinter),
            "seed": splinter[0],
            "transfers": transfers,
        })

        next_cluster_id += 2

    return clusters, split_history


def cluster_labels(clusters, n_observations):
    """
    Convert a collection of clusters into observation labels.

    Input:
        clusters: Dictionary of active clusters.
        n_observations: Number of observations.

    Output:
        labels: Cluster label for each observation.
    """
    labels = np.empty(
        n_observations,
        dtype=int,
    )

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda members: min(members),
    )

    for label, members in enumerate(ordered_clusters):
        labels[members] = label

    return labels


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Partition Diagnostics
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def within_cluster_sum_squares(x, cluster):
    """
    Compute within-cluster sum of squares.

    Input:
        x: Observation matrix.
        cluster: Observation indices in the cluster.

    Output:
        Within-cluster sum of squares.
    """
    centroid = np.mean(
        x[cluster],
        axis=0,
    )

    deviations = x[cluster] - centroid

    return np.sum(deviations**2)


def partition_wcss(x, clusters):
    """
    Compute total WCSS for a divisive partition.

    Input:
        x: Observation matrix.
        clusters: Dictionary of active clusters.

    Output:
        Total within-cluster sum of squares.
    """
    return sum(
        within_cluster_sum_squares(x, cluster)
        for cluster in clusters.values()
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


def plot_clusters(x, labels):
    """
    Plot clusters obtained by divisive hierarchical clustering.

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
        title="Divisive Hierarchical Clustering",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_split_history(x, split_history):
    """
    Plot the partition produced after each divisive split.

    Input:
        x: Observation matrix.
        split_history: Detailed information for each split.

    Output:
        Displays the successive divisive partitions.
    """
    clusters = {
        0: list(range(len(x)))
    }

    for split in split_history:
        parent = split["parent"]

        parent_members = clusters.pop(parent)

        remainder = [
            observation
            for observation in parent_members
            if observation != split["seed"]
        ]

        splinter = [split["seed"]]

        for transfer in split["transfers"]:
            observation = transfer["observation"]
            remainder.remove(observation)
            splinter.append(observation)

        clusters[split["remainder_id"]] = remainder
        clusters[split["splinter_id"]] = splinter

        labels = cluster_labels(
            clusters,
            len(x),
        )

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
            title=(
                f"Divisive Clustering — "
                f"{len(clusters)} Clusters"
            ),
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
    Generate data and apply divisive hierarchical clustering.

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

    distances = pairwise_distances(x)

    initial_cluster = list(range(len(x)))

    initial_wcss = within_cluster_sum_squares(
        x,
        initial_cluster,
    )

    clusters, split_history = divisive_clustering(
        x,
        n_clusters,
    )

    labels = cluster_labels(
        clusters,
        len(x),
    )

    final_wcss = partition_wcss(
        x,
        clusters,
    )

    # Main Results

    print(separator)
    print("Divisive Hierarchical Clustering")
    print(separator)
    print()

    print(f"Observations:          {len(x)}")
    print(f"Features:              {x.shape[1]}")
    print(f"True groups:           {len(np.unique(true_labels))}")
    print(f"Selected clusters:     {n_clusters}")
    print("Initial clusters:      1")
    print(f"Final clusters:        {len(clusters)}")

    # Initial Partition

    print()
    print(separator)
    print("Initial Partition")
    print(separator)
    print()

    print(f"Cluster size:          {len(x)}")
    print(
        f"Cluster diameter:      "
        f"{cluster_diameter(initial_cluster, distances):.6f}"
    )
    print(f"Initial WCSS:          {initial_wcss:.6f}")

    # Divisive Sequence

    print()
    print(separator)
    print("Divisive Sequence")
    print(separator)

    for split in split_history:
        print()
        print(f"Split {split['step']}")
        print(
            f"Parent cluster:        "
            f"{split['parent']}"
        )
        print(
            f"Parent size:           "
            f"{split['parent_size']}"
        )
        print(
            f"Parent diameter:       "
            f"{split['diameter']:.6f}"
        )
        print(
            f"Splinter seed:         "
            f"{split['seed']}"
        )
        print(
            f"Transfers:             "
            f"{len(split['transfers'])}"
        )
        print(
            f"Remainder size:        "
            f"{split['remainder_size']}"
        )
        print(
            f"Splinter size:         "
            f"{split['splinter_size']}"
        )

    # Transfer Details

    print()
    print(separator)
    print("Transfer Details")
    print(separator)

    for split in split_history:
        print()
        print(f"Split {split['step']}")

        if not split["transfers"]:
            print("No additional transfers.")
            continue

        for transfer in split["transfers"]:
            print(
                f"Observation "
                f"{transfer['observation']:2d}: "
                f"delta={transfer['score']:.6f}"
            )

    # Final Partition

    print()
    print(separator)
    print(f"Partition — {n_clusters} Clusters")
    print(separator)
    print()

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda members: min(members),
    )

    for label, cluster in enumerate(ordered_clusters):
        diameter = cluster_diameter(
            cluster,
            distances,
        )

        print(
            f"Cluster {label + 1}:             "
            f"{len(cluster)} observations, "
            f"diameter={diameter:.6f}"
        )

    # Partition Diagnostics

    print()
    print(separator)
    print("Partition Diagnostics")
    print(separator)
    print()

    print(f"Initial WCSS:          {initial_wcss:.6f}")
    print(f"Final WCSS:            {final_wcss:.6f}")
    print(
        f"WCSS reduction:        "
        f"{initial_wcss - final_wcss:.6f}"
    )

    # Visualisation

    plot_data(
        x,
        true_labels,
    )

    plot_split_history(
        x,
        split_history,
    )

    plot_clusters(
        x,
        labels,
    )


if __name__ == "__main__":
    main()