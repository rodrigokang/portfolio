
"""
Principal Component Analysis — Singular Value Decomposition.

NumPy: explicit PCA from the singular value decomposition.
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

def compute_pca_svd(X):
    """
    Compute PCA using the singular value decomposition.

    Input:
        X: Data matrix with observations in rows.

    Output:
        mean: Sample mean of each variable.
        X_centered: Centred data matrix.
        U: Left singular vectors.
        singular_values: Singular values in descending order.
        Vt: Transposed right singular vectors.
        eigenvalues: Principal component variances.
        scores: Principal component scores.
    """
    n = X.shape[0]

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    U, singular_values, Vt = np.linalg.svd(
        X_centered, full_matrices=False
    )

    eigenvalues = singular_values**2 / (n - 1)

    scores = X_centered @ Vt.T

    return (
        mean,
        X_centered,
        U,
        singular_values,
        Vt,
        eigenvalues,
        scores,
    )


def reconstruct_data(U, singular_values, Vt, mean, q):
    """
    Reconstruct observations using the first q components.

    Input:
        U: Left singular vectors.
        singular_values: Singular values.
        Vt: Transposed right singular vectors.
        mean: Sample mean of each variable.
        q: Number of retained components.

    Output:
        X_reconstructed: Reconstructed data matrix.
    """
    U_reduced = U[:, :q]
    singular_reduced = singular_values[:q]
    Vt_reduced = Vt[:q, :]

    X_reconstructed = (
        (U_reduced * singular_reduced) @ Vt_reduced
        + mean
    )

    return X_reconstructed


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_principal_directions(X, mean, eigenvalues, Vt):
    """
    Plot observations and principal directions from SVD.

    Input:
        X: Original data matrix.
        mean: Sample mean of each variable.
        eigenvalues: Principal component variances.
        Vt: Transposed right singular vectors.

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
            2 * np.sqrt(eigenvalues[j]) * Vt[j, :]
        )

        ax.quiver(
            *mean, *direction,
            color=colors[j], linewidth=2.5,
            arrow_length_ratio=0.12,
            label=f"PC{j + 1}"
        )

    ax.set(
        title="Principal Component Directions — SVD",
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
        eigenvalues: Principal component variances.

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
        title="Explained Variance — SVD",
        xlabel="Principal Component",
        ylabel="Explained Variance (%)",
    )

    ax.set_xticks(components)
    ax.set_ylim(0, 105)

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    plt.show()


def plot_reconstruction_errors(errors, explained):
    """
    Plot reconstruction error against retained components.

    Input:
        errors: Squared reconstruction errors.
        explained: Cumulative explained variance.

    Output:
        Displays the reconstruction error figure.
    """
    components = np.arange(1, len(errors) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        components, errors,
        marker="o", linewidth=1.8,
        label="Reconstruction error"
    )

    ax.set(
        title="PCA Reconstruction Error — SVD",
        xlabel="Retained Components",
        ylabel="Squared Reconstruction Error",
    )

    ax.set_xticks(components)
    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.2)

    for q, error, variance in zip(
        components, errors, explained
    ):
        ax.annotate(
            f"{100 * variance:.1f}%",
            (q, error),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Compute PCA using SVD and verify its properties.

    Input:
        None.

    Output:
        Prints numerical results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n = 1000
    separator = "<>" * 36

    X = generate_data(n, rng)

    (
        mean,
        X_centered,
        U,
        singular_values,
        Vt,
        eigenvalues,
        scores,
    ) = compute_pca_svd(X)

    n, m = X.shape
    rank = np.linalg.matrix_rank(X_centered)

    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)

    # Main Results

    print(separator)
    print("Principal Component Analysis — SVD")
    print(separator)
    print()

    print(f"Observations:          {n}")
    print(f"Variables:             {m}")
    print(f"Random seed:           42")
    print(f"Centred matrix rank:   {rank}")

    # Singular Value Decomposition

    print()
    print(separator)
    print("Singular Value Decomposition")
    print(separator)
    print()

    print(f"Shape of U:           {U.shape}")
    print(f"Shape of Vt:          {Vt.shape}")

    print()
    print("Singular values:")
    print(singular_values)

    print()
    print("Right singular vectors (rows of Vt):")
    print(Vt)

    # Principal Components

    print()
    print(separator)
    print("Principal Components")
    print(separator)
    print()

    print("Eigenvalues:")
    print(eigenvalues)

    print()
    print("Eigenvectors (columns of V):")
    print(Vt.T)

    print()
    print("First five component scores:")
    print(scores[:5])

    # Explained Variance

    print()
    print(separator)
    print("Explained Variance")
    print(separator)
    print()

    for j in range(len(eigenvalues)):
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

    identity = np.eye(len(singular_values))

    orthogonal_u = np.allclose(
        U.T @ U, identity
    )

    orthogonal_v = np.allclose(
        Vt @ Vt.T, identity
    )

    svd_reconstruction = (
        (U * singular_values) @ Vt
    )

    decomposition_agrees = np.allclose(
        X_centered, svd_reconstruction
    )

    scores_agree = np.allclose(
        scores, U * singular_values
    )

    covariance = X_centered.T @ X_centered / (n - 1)

    score_covariance = scores.T @ scores / (n - 1)

    diagonal_covariance = np.allclose(
        score_covariance, np.diag(eigenvalues)
    )

    total_variance = np.isclose(
        np.trace(covariance),
        np.sum(eigenvalues)
    )

    print(f"Orthonormal U:        {orthogonal_u}")
    print(f"Orthonormal V:        {orthogonal_v}")
    print(f"SVD reconstruction:   {decomposition_agrees}")
    print(f"Scores = U @ D:       {scores_agree}")
    print(f"Diagonal score cov.:  {diagonal_covariance}")
    print(f"Trace = eigenvalue sum: {total_variance}")

    # Eigendecomposition Comparison

    reference_values, reference_vectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(reference_values)[::-1]

    reference_values = reference_values[order]
    reference_vectors = reference_vectors[:, order]

    # Eigenvector signs are arbitrary.
    aligned_vectors = Vt.T.copy()

    for j in range(len(eigenvalues)):
        if np.dot(
            aligned_vectors[:, j],
            reference_vectors[:, j]
        ) < 0:
            aligned_vectors[:, j] *= -1

    eigenvalues_agree = np.allclose(
        eigenvalues, reference_values
    )

    eigenvectors_agree = np.allclose(
        aligned_vectors, reference_vectors
    )

    print()
    print(separator)
    print("Eigendecomposition Comparison")
    print(separator)
    print()

    print("SVD eigenvalues:")
    print(eigenvalues)

    print()
    print("Eigendecomposition eigenvalues:")
    print(reference_values)

    print()
    print(f"Eigenvalues agree:    {eigenvalues_agree}")
    print(f"Eigenvectors agree:   {eigenvectors_agree}")

    # Reconstruction

    print()
    print(separator)
    print("Reconstruction")
    print(separator)
    print()

    errors = []

    for q in range(1, len(singular_values) + 1):
        X_reconstructed = reconstruct_data(
            U, singular_values, Vt, mean, q
        )

        error = np.linalg.norm(
            X - X_reconstructed, ord="fro"
        ) ** 2

        theoretical_error = np.sum(
            singular_values[q:]**2
        )

        errors_agree = np.isclose(
            error, theoretical_error, atol=1e-8
        )

        errors.append(error)

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
        X, mean, eigenvalues, Vt
    )

    plot_explained_variance(eigenvalues)

    plot_reconstruction_errors(
        errors, cumulative
    )


if __name__ == "__main__":
    main()
