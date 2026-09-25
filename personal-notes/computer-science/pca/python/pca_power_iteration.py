
"""
Principal Component Analysis — Power Iteration.

NumPy: explicit covariance and iterative eigenvector estimation.
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
# Power Iteration
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def power_iteration(covariance, initial, tolerance=1e-10,
                    max_iterations=1000):
    """
    Estimate the dominant eigenvector by Power Iteration.

    Input:
        covariance: Sample covariance matrix.
        initial: Initial direction.
        tolerance: Eigenvalue residual tolerance.
        max_iterations: Maximum number of updates.

    Output:
        eigenvalue: Estimated dominant eigenvalue.
        eigenvector: Estimated dominant eigenvector.
        history: Residual norm after each update.
        converged: Whether the tolerance was reached.
    """
    eigenvector = np.asarray(initial, dtype=float).copy()
    eigenvector /= np.linalg.norm(eigenvector)

    history = []

    for _ in range(max_iterations):
        direction = covariance @ eigenvector
        norm = np.linalg.norm(direction)

        if norm == 0:
            break

        eigenvector = direction / norm

        eigenvalue = eigenvector @ covariance @ eigenvector

        residual = np.linalg.norm(
            covariance @ eigenvector - eigenvalue * eigenvector
        )

        history.append(residual)

        if residual <= tolerance:
            break

    eigenvalue = eigenvector @ covariance @ eigenvector

    residual = np.linalg.norm(
        covariance @ eigenvector - eigenvalue * eigenvector
    )

    converged = residual <= tolerance

    return eigenvalue, eigenvector, np.asarray(history), converged


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_convergence(history, tolerance):
    """
    Plot the eigenvalue residual during Power Iteration.

    Input:
        history: Residual norm after each update.
        tolerance: Convergence tolerance.

    Output:
        Displays the convergence figure.
    """
    iterations = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        iterations, history,
        color="tab:blue", linewidth=1.8,
        label="Eigenvalue residual"
    )

    ax.axhline(
        tolerance, color="tab:red", linestyle="--",
        linewidth=1.5, label="Tolerance"
    )

    ax.set(
        title="Power Iteration Convergence",
        xlabel="Iteration",
        ylabel="Residual Norm",
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


def plot_principal_direction(X, mean, eigenvalue, eigenvector):
    """
    Plot observations and the estimated principal direction.

    Input:
        X: Original data matrix.
        mean: Sample mean of each variable.
        eigenvalue: Estimated dominant eigenvalue.
        eigenvector: Estimated dominant eigenvector.

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

    direction = 2 * np.sqrt(eigenvalue) * eigenvector

    ax.quiver(
        *mean, *direction,
        color="tab:red", linewidth=2.5,
        arrow_length_ratio=0.12,
        label="PC1"
    )

    ax.set(
        title="First Principal Component — Power Iteration",
        xlabel="$X_1$",
        ylabel="$X_2$",
        zlabel="$X_3$",
    )

    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Estimate the first principal component by Power Iteration.

    Input:
        None.

    Output:
        Prints numerical results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n = 1000
    tolerance = 1e-10
    max_iterations = 1000

    separator = "<>" * 36

    X = generate_data(n, rng)

    n, m = X.shape

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    covariance = X_centered.T @ X_centered / (n - 1)

    initial = np.ones(m)
    initial /= np.linalg.norm(initial)

    eigenvalue, eigenvector, history, converged = power_iteration(
        covariance, initial, tolerance, max_iterations
    )

    # Main Results

    print(separator)
    print("Principal Component Analysis — Power Iteration")
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

    # Power Iteration Results

    print()
    print(separator)
    print("Power Iteration Results")
    print(separator)
    print()

    print("Initial direction:")
    print(initial)

    print()
    print("Estimated eigenvector:")
    print(eigenvector)

    print()
    print(f"Estimated eigenvalue: {eigenvalue:.10f}")

    # Convergence

    print()
    print(separator)
    print("Convergence")
    print(separator)
    print()

    residual = np.linalg.norm(
        covariance @ eigenvector - eigenvalue * eigenvector
    )

    print(f"Tolerance:            {tolerance:.2e}")
    print(f"Maximum iterations:   {max_iterations}")
    print(f"Iterations:           {len(history)}")
    print(f"Final residual:       {residual:.10e}")
    print(f"Converged:            {converged}")

    # Eigendecomposition Comparison

    reference_values, reference_vectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(reference_values)[::-1]

    reference_values = reference_values[order]
    reference_vectors = reference_vectors[:, order]

    reference_value = reference_values[0]
    reference_vector = reference_vectors[:, 0]

    # Eigenvector signs are arbitrary.
    if np.dot(eigenvector, reference_vector) < 0:
        reference_vector = -reference_vector

    eigenvalue_error = abs(eigenvalue - reference_value)

    direction_error = np.linalg.norm(
        eigenvector - reference_vector
    )

    print()
    print(separator)
    print("Eigendecomposition Comparison")
    print(separator)
    print()

    print(f"Power Iteration:       {eigenvalue:.10f}")
    print(f"Eigendecomposition:    {reference_value:.10f}")
    print(f"Eigenvalue error:      {eigenvalue_error:.10e}")

    print()
    print("Reference eigenvector:")
    print(reference_vector)

    print()
    print(f"Direction error:      {direction_error:.10e}")

    print(
        "Eigenvalues agree:    "
        f"{np.isclose(eigenvalue, reference_value)}"
    )

    print(
        "Eigenvectors agree:   "
        f"{np.allclose(eigenvector, reference_vector)}"
    )

    # Explained Variance

    explained = eigenvalue / np.sum(reference_values)

    print()
    print(separator)
    print("Explained Variance")
    print(separator)
    print()

    print(f"Total variance:       {np.trace(covariance):.6f}")
    print(f"PC1 eigenvalue:       {eigenvalue:.6f}")
    print(f"PC1 explained:        {100 * explained:.4f}%")

    # Reconstruction

    scores = X_centered @ eigenvector

    X_reconstructed = (
        np.outer(scores, eigenvector) + mean
    )

    error = np.linalg.norm(
        X - X_reconstructed, ord="fro"
    ) ** 2

    theoretical_error = (
        (n - 1) * np.sum(reference_values[1:])
    )

    errors_agree = np.isclose(
        error, theoretical_error, atol=1e-8
    )

    print()
    print(separator)
    print("Reconstruction")
    print(separator)
    print()

    print(f"Retained components:   1")
    print(f"Reconstruction error:  {error:.8f}")
    print(f"Theoretical error:     {theoretical_error:.8f}")
    print(f"Errors agree:          {errors_agree}")

    # Visualisation

    plot_convergence(history, tolerance)

    plot_principal_direction(
        X, mean, eigenvalue, eigenvector
    )


if __name__ == "__main__":
    main()
