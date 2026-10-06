
"""
Principal Component Analysis — Eigendecomposition.

NumPy: explicit covariance, projections and reconstruction.
Matplotlib: figures displayed, not saved.
"""


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import numpy as np
import matplotlib.pyplot as plt


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Synthetic Data
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def generate_data(n, rng):
    """
    Generate three correlated numerical variables.

    Input:
        n: Number of observations.
        rng: NumPy random number generator.

    Output:
        X: Synthetic data matrix.
    """
    u = rng.normal(0, 1, n)
    v = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 3))

    x1 = 2.0 * u + 0.2 * noise[:, 0]
    x2 = 1.5 * u + 0.5 * v + 0.2 * noise[:, 1]
    x3 = v + 0.2 * noise[:, 2]

    X = np.column_stack((x1, x2, x3))

    return X


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Principal Component Analysis
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def compute_pca(X):
    """
    Compute PCA from the sample covariance matrix.

    Input:
        X: Data matrix with observations in rows.

    Output:
        mean: Sample mean of each variable.
        X_centered: Centred data matrix.
        covariance: Sample covariance matrix.
        eigenvalues: Eigenvalues in descending order.
        eigenvectors: Corresponding principal directions.
        scores: Principal component scores.
    """
    n = X.shape[0]

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    covariance = X_centered.T @ X_centered / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scores = X_centered @ eigenvectors

    return (
        mean,
        X_centered,
        covariance,
        eigenvalues,
        eigenvectors,
        scores,
    )


def reconstruct_data(scores, eigenvectors, mean, q):
    """
    Reconstruct observations using the first q components.

    Input:
        scores: Principal component scores.
        eigenvectors: Principal directions.
        mean: Sample mean of each variable.
        q: Number of retained components.

    Output:
        X_reconstructed: Reconstructed data matrix.
    """
    scores_reduced = scores[:, :q]
    directions = eigenvectors[:, :q]

    X_reconstructed = scores_reduced @ directions.T + mean

    return X_reconstructed


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_principal_directions(X, mean, eigenvalues, eigenvectors):
    """
    Plot observations and the three principal directions.

    Input:
        X: Original data matrix.
        mean: Sample mean of each variable.
        eigenvalues: Eigenvalues in descending order.
        eigenvectors: Corresponding principal directions.

    Output:
        Displays a three-dimensional PCA figure.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        s=10, alpha=0.2, label="Observations"
    )

    ax.scatter(
        *mean, color="black", s=45,
        label="Sample mean"
    )

    colors = ("tab:red", "tab:green", "tab:orange")

    for j in range(X.shape[1]):
        direction = (
            2 * np.sqrt(eigenvalues[j]) * eigenvectors[:, j]
        )

        ax.quiver(
            *mean, *direction,
            color=colors[j], linewidth=2.5,
            arrow_length_ratio=0.12,
            label=f"PC{j + 1}"
        )

    ax.set(
        title="Principal Component Directions",
        xlabel="$X_1$",
        ylabel="$X_2$",
        zlabel="$X_3$",
    )

    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_explained_variance(eigenvalues):
    """
    Plot individual and cumulative explained variance.

    Input:
        eigenvalues: Eigenvalues in descending order.

    Output:
        Displays the explained variance figure.
    """
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)

    components = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        components, 100 * explained,
        alpha=0.7, label="Individual"
    )

    ax.plot(
        components, 100 * cumulative,
        marker="o", linewidth=2,
        label="Cumulative"
    )

    ax.set(
        title="Explained Variance",
        xlabel="Principal Component",
        ylabel="Explained Variance (%)",
    )

    ax.set_xticks(components)
    ax.set_ylim(0, 105)

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Compute PCA and verify its mathematical properties.

    Input:
        None.

    Output:
        Prints PCA results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n = 1000
    separator = "<>" * 36

    X = generate_data(n, rng)

    (
        mean,
        X_centered,
        covariance,
        eigenvalues,
        eigenvectors,
        scores,
    ) = compute_pca(X)

    n, m = X.shape

    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)

    # Main Results

    print(separator)
    print("Principal Component Analysis — Eigendecomposition")
    print(separator)
    print()

    print(f"Observations:          {n}")
    print(f"Variables:             {m}")
    print(f"Random seed:           42")

    # Covariance Matrix

    print()
    print(separator)
    print("Covariance Matrix")
    print(separator)
    print()

    print("Sample mean:")
    print(mean)

    print()
    print("Sample covariance matrix:")
    print(covariance)

    # Principal Components

    print()
    print(separator)
    print("Principal Components")
    print(separator)
    print()

    print("Eigenvalues:")
    print(eigenvalues)

    print()
    print("Eigenvectors (columns):")
    print(eigenvectors)

    print()
    print("First five component scores:")
    print(scores[:5])

    # Explained Variance

    print()
    print(separator)
    print("Explained Variance")
    print(separator)
    print()

    for j in range(m):
        print(
            f"PC{j + 1}:  "
            f"individual={100 * explained[j]:8.4f}%  "
            f"cumulative={100 * cumulative[j]:8.4f}%"
        )

    # Numerical Verification

    print()
    print(separator)
    print("Numerical Verification")
    print(separator)
    print()

    identity = np.eye(m)
    score_covariance = scores.T @ scores / (n - 1)

    orthogonal = np.allclose(
        eigenvectors.T @ eigenvectors,
        identity
    )

    diagonal_covariance = np.allclose(
        score_covariance,
        np.diag(eigenvalues)
    )

    total_variance = np.isclose(
        np.trace(covariance),
        np.sum(eigenvalues)
    )

    centered = np.allclose(
        np.mean(X_centered, axis=0),
        0
    )

    print(f"Centred variables:     {centered}")
    print(f"Orthonormal vectors:   {orthogonal}")
    print(f"Diagonal score cov.:   {diagonal_covariance}")
    print(f"Trace = eigenvalue sum:{total_variance:>6}")

    print()
    print(f"Total variance:        {np.trace(covariance):.6f}")
    print(f"Eigenvalue sum:        {np.sum(eigenvalues):.6f}")

    # Reconstruction

    print()
    print(separator)
    print("Reconstruction")
    print(separator)
    print()

    for q in range(1, m + 1):
        X_reconstructed = reconstruct_data(
            scores, eigenvectors, mean, q
        )

        error = np.linalg.norm(
            X - X_reconstructed, ord="fro"
        ) ** 2

        theoretical_error = (
            (n - 1) * np.sum(eigenvalues[q:])
        )

        errors_agree = np.isclose(
            error, theoretical_error, atol=1e-10
        )

        print(f"Retained components:   {q}")
        print(
            f"Explained variance:    "
            f"{100 * cumulative[q - 1]:.4f}%"
        )
        print(f"Reconstruction error:  {error:.8f}")
        print(f"Theoretical error:     {theoretical_error:.8f}")
        print(f"Errors agree:          {errors_agree}")
        print()

    # Visualisation

    plot_principal_directions(
        X, mean, eigenvalues, eigenvectors
    )

    plot_explained_variance(eigenvalues)


if __name__ == "__main__":
    main()
