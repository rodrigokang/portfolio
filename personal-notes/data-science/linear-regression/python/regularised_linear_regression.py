"""
Regularised Linear Regression — Computational Implementation.

NumPy: explicit Ridge, Lasso and Elastic Net estimators.
SciPy: not required; no statistical inference is applied to penalised fits.
Matplotlib: figures displayed, not saved.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import matplotlib.pyplot as plt
import numpy as np

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Synthetic Data
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def generate_data(n, coefficients, noise_std, rng):
    """
    Generate correlated predictors and a linear response.

    Input:
        Sample size, true coefficients, noise standard deviation and RNG.

    Output:
        Predictor matrix and response vector.
    """
    x1 = rng.normal(size=n)
    x2 = 0.85 * x1 + np.sqrt(1 - 0.85**2) * rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = 0.65 * x3 + np.sqrt(1 - 0.65**2) * rng.normal(size=n)
    x5 = rng.normal(size=n)
    x6 = rng.normal(size=n)
    x = np.column_stack((x1, x2, x3, x4, x5, x6))
    y = coefficients[0] + x @ coefficients[1:]
    y += rng.normal(scale=noise_std, size=n)
    return x, y


def train_test_split(x, y, test_fraction, rng):
    """
    Randomly split observations into training and test sets.

    Input:
        Predictors, response, test fraction and RNG.

    Output:
        Training and test predictors and responses.
    """
    indices = rng.permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    if not 0 < n_test < len(y):
        raise ValueError("Both partitions must contain observations.")
    test = indices[:n_test]
    train = indices[n_test:]
    return x[train], x[test], y[train], y[test]


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Standardisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def fit_standardisation(x):
    """
    Estimate predictor scaling from training observations only.

    Input:
        Training predictor matrix.

    Output:
        Column means and population standard deviations.
    """
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0, ddof=0)
    if np.any(scales == 0):
        raise ValueError("Predictors must have nonzero variance.")
    return means, scales


def standardise(x, means, scales):
    """
    Apply training-set predictor scaling.

    Input:
        Predictor matrix, training means and training scales.

    Output:
        Standardised predictor matrix.
    """
    return (x - means) / scales


def original_coefficients(intercept, slopes, means, scales):
    """
    Express a standardised model in original predictor units.

    Input:
        Standardised intercept and slopes, means and scales.

    Output:
        Intercept and slopes in original units.
    """
    original_slopes = slopes / scales
    original_intercept = intercept - means @ original_slopes
    return np.r_[original_intercept, original_slopes]


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Objective Function and Estimators
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def objective_function(x, y, intercept, slopes, penalty, l1_ratio):
    """
    Evaluate mean-scaled RSS plus Elastic Net penalty.

    Input:
        Predictors, response, parameters, penalty and L1 ratio.

    Output:
        Penalised objective value.
    """
    residuals = y - intercept - x @ slopes
    loss = (residuals @ residuals) / (2 * len(y))
    l1 = l1_ratio * np.sum(np.abs(slopes))
    l2 = (1 - l1_ratio) * (slopes @ slopes) / 2
    return float(loss + penalty * (l1 + l2))


def ordinary_least_squares(x, y):
    """
    Estimate unpenalised coefficients using least squares.

    Input:
        Predictor matrix and response.

    Output:
        Intercept and slopes.
    """
    design = np.column_stack((np.ones(len(y)), x))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("OLS requires full column rank.")
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def ridge_regression(x, y, penalty):
    """
    Fit Ridge with an unpenalised intercept by linear solve.

    Input:
        Standardised predictors, response and nonnegative penalty.

    Output:
        Intercept and slopes.
    """
    if penalty < 0:
        raise ValueError("Penalty must be nonnegative.")
    n, p = x.shape
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y)
    centered = x - x_mean
    slopes = np.linalg.solve(
        centered.T @ centered / n + penalty * np.eye(p),
        centered.T @ (y - y_mean) / n,
    )
    return np.r_[y_mean - x_mean @ slopes, slopes]


def soft_threshold(value, threshold):
    """
    Apply the scalar soft-thresholding operator.

    Input:
        Scalar value and nonnegative threshold.

    Output:
        Thresholded scalar.
    """
    return np.sign(value) * max(abs(value) - threshold, 0.0)


def coordinate_descent(x, y, penalty, l1_ratio, tolerance=1e-8,
                       max_iterations=10000):
    """
    Fit Lasso or Elastic Net using cyclic coordinate descent.

    Input:
        Standardised predictors, response, penalty, L1 ratio and limits.

    Output:
        Coefficients, objective history and convergence flag.
    """
    if penalty < 0 or not 0 <= l1_ratio <= 1:
        raise ValueError("Penalty and L1 ratio are outside their domains.")
    n, p = x.shape
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y)
    centered = x - x_mean
    target = y - y_mean
    squared_norms = np.sum(centered**2, axis=0) / n
    if np.any(squared_norms == 0):
        raise ValueError("Predictors must have nonzero variance.")

    slopes = np.zeros(p)
    residuals = target.copy()
    history = [objective_function(x, y, y_mean, slopes,
                                  penalty, l1_ratio)]
    converged = False

    for _ in range(max_iterations):
        previous = slopes.copy()
        for j in range(p):
            residuals += centered[:, j] * slopes[j]
            correlation = centered[:, j] @ residuals / n
            slopes[j] = soft_threshold(
                correlation, penalty * l1_ratio
            ) / (squared_norms[j] + penalty * (1 - l1_ratio))
            residuals -= centered[:, j] * slopes[j]

        intercept = y_mean - x_mean @ slopes
        history.append(objective_function(
            x, y, intercept, slopes, penalty, l1_ratio
        ))
        if np.max(np.abs(slopes - previous)) <= tolerance:
            converged = True
            break

    intercept = y_mean - x_mean @ slopes
    return np.r_[intercept, slopes], np.asarray(history), converged


def predict(x, coefficients):
    """
    Compute predictions from an intercept and slopes.

    Input:
        Predictor matrix and coefficient vector.

    Output:
        Predicted response.
    """
    return coefficients[0] + x @ coefficients[1:]


def regression_metrics(y, fitted):
    """
    Calculate RSS, mean squared error and R-squared.

    Input:
        Observed and fitted responses.

    Output:
        Dictionary of predictive metrics.
    """
    residuals = y - fitted
    rss = float(residuals @ residuals)
    tss = float(np.sum((y - np.mean(y))**2))
    return {
        "rss": rss,
        "mse": rss / len(y),
        "rmse": np.sqrt(rss / len(y)),
        "r_squared": 1 - rss / tss if tss > 0 else np.nan,
    }


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Cross-Validation and Regularisation Paths
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def fit_model(x, y, method, penalty, l1_ratio=0.5):
    """
    Fit a specified regularised estimator.

    Input:
        Standardised predictors, response, method and penalties.

    Output:
        Coefficient vector and convergence flag.
    """
    if method == "Ridge":
        return ridge_regression(x, y, penalty), True
    if method == "Lasso":
        coefficients, _, converged = coordinate_descent(
            x, y, penalty, 1.0
        )
        return coefficients, converged
    if method == "Elastic Net":
        coefficients, _, converged = coordinate_descent(
            x, y, penalty, l1_ratio
        )
        return coefficients, converged
    raise ValueError(f"Unknown method: {method}")


def cross_validate(x, y, method, penalties, folds, l1_ratio=0.5):
    """
    Evaluate penalties with fold-specific standardisation.

    Input:
        Training predictors, response, method, grid, folds and L1 ratio.

    Output:
        Mean validation MSE, standard error and convergence flags.
    """
    errors = np.empty((len(penalties), len(folds)))
    convergence = np.ones(len(penalties), dtype=bool)
    for fold_index, validation in enumerate(folds):
        train = np.ones(len(y), dtype=bool)
        train[validation] = False
        means, scales = fit_standardisation(x[train])
        x_train = standardise(x[train], means, scales)
        x_valid = standardise(x[validation], means, scales)
        for i, penalty in enumerate(penalties):
            coefficients, converged = fit_model(
                x_train, y[train], method, penalty, l1_ratio
            )
            errors[i, fold_index] = np.mean(
                (y[validation] - predict(x_valid, coefficients))**2
            )
            convergence[i] &= converged
    return (np.mean(errors, axis=1),
            np.std(errors, axis=1, ddof=1) / np.sqrt(len(folds)),
            convergence)


def regularisation_path(x, y, method, penalties, l1_ratio=0.5):
    """
    Fit standardised coefficients along a penalty grid.

    Input:
        Standardised predictors, response, method, grid and L1 ratio.

    Output:
        Matrix of coefficients and convergence flags.
    """
    coefficients = np.empty((len(penalties), x.shape[1] + 1))
    convergence = np.empty(len(penalties), dtype=bool)
    for i, penalty in enumerate(penalties):
        coefficients[i], convergence[i] = fit_model(
            x, y, method, penalty, l1_ratio
        )
    return coefficients, convergence


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_cross_validation(penalties, cv_results, selected):
    """
    Display validation MSE and selected penalties.

    Input:
        Penalty grid, CV results and selected penalty dictionary.

    Output:
        Displays cross-validation curves.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, (mean, standard_error, _) in cv_results.items():
        ax.plot(penalties, mean, marker="o", markersize=3, label=method)
        ax.fill_between(penalties, mean - standard_error,
                        mean + standard_error, alpha=0.12)
        ax.axvline(selected[method], linestyle=":", alpha=0.45)
    ax.set(xscale="log", title="Cross-Validation",
           xlabel="Penalty (lambda)", ylabel="Validation MSE")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_regularisation_paths(penalties, paths):
    """
    Display standardised slope paths for each method.

    Input:
        Penalty grid and dictionary of coefficient paths.

    Output:
        Displays coefficient paths.
    """
    for method, coefficients in paths.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for j in range(1, coefficients.shape[1]):
            ax.plot(penalties, coefficients[:, j], label=f"x{j}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xscale="log", title=f"{method} — Coefficient Paths",
               xlabel="Penalty (lambda)",
               ylabel="Standardised coefficient")
        ax.legend(frameon=False, ncol=3)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.show()


def plot_predictions(y_test, predictions):
    """
    Compare test responses with model predictions.

    Input:
        Test responses and dictionary of model predictions.

    Output:
        Displays observed-versus-predicted figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, fitted in predictions.items():
        ax.scatter(y_test, fitted, s=20, alpha=0.35, label=method)
    limits = (min(y_test.min(), *(v.min() for v in predictions.values())),
              max(y_test.max(), *(v.max() for v in predictions.values())))
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.set(title="Test Set — Observed vs Predicted",
           xlabel="Observed response", ylabel="Predicted response")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Fit, tune and compare OLS, Ridge, Lasso and Elastic Net.

    Input:
        None.

    Output:
        Prints estimation results and displays figures.
    """
    rng = np.random.default_rng(seed=42)
    n = 1000
    true_coefficients = np.array([1.0, 2.0, 0.0, -1.5, 0.0, 0.8, 0.0])
    noise_std = 1.0
    test_fraction = 0.2
    n_folds = 5
    l1_ratio = 0.5
    penalties = np.logspace(-3, 1, 35)
    separator = "<>" * 36

    x, y = generate_data(n, true_coefficients, noise_std, rng)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_fraction, rng
    )
    means, scales = fit_standardisation(x_train)
    z_train = standardise(x_train, means, scales)
    z_test = standardise(x_test, means, scales)

    ols = ordinary_least_squares(z_train, y_train)
    folds = np.array_split(rng.permutation(len(y_train)), n_folds)
    methods = ("Ridge", "Lasso", "Elastic Net")
    cv_results = {}
    selected = {}
    models = {"OLS": ols}
    paths = {}

    for method in methods:
        mean, standard_error, converged = cross_validate(
            x_train, y_train, method, penalties, folds, l1_ratio
        )
        cv_results[method] = (mean, standard_error, converged)
        if not np.all(converged):
            raise RuntimeError(f"Cross-validation did not converge: {method}")
        best = int(np.argmin(mean))
        selected[method] = penalties[best]
        models[method], converged = fit_model(
            z_train, y_train, method, penalties[best], l1_ratio
        )
        if not converged:
            raise RuntimeError(f"Final fit did not converge: {method}")
        paths[method], path_converged = regularisation_path(
            z_train, y_train, method, penalties, l1_ratio
        )
        if not np.all(path_converged):
            raise RuntimeError(f"Coefficient path did not converge: {method}")

    print(separator)
    print("Regularised Linear Regression")
    print(separator)
    print()
    print(f"Observations:         {n}")
    print(f"Training observations: {len(y_train)}")
    print(f"Test observations:    {len(y_test)}")
    print(f"Predictors:           {x.shape[1]}")
    print(f"Cross-validation:     {n_folds} folds")
    print(f"Elastic Net L1 ratio: {l1_ratio:.2f}")

    print()
    print(separator)
    print("Regularisation and Cross-Validation")
    print(separator)
    print()
    for method in methods:
        best = np.argmin(cv_results[method][0])
        print(f"{method:12s} lambda={selected[method]:.6g} "
              f"CV MSE={cv_results[method][0][best]:.6f} "
              f"SE={cv_results[method][1][best]:.6f}")

    print()
    print(separator)
    print("Coefficient Estimates (Original Units)")
    print(separator)
    print()
    print(f"{'Parameter':12s} {'True':>10s} "
          + " ".join(f"{name:>12s}" for name in models))
    original = {
        name: original_coefficients(
            coefficients[0], coefficients[1:], means, scales
        ) for name, coefficients in models.items()
    }
    for j in range(len(true_coefficients)):
        label = "Intercept" if j == 0 else f"x{j}"
        print(f"{label:12s} {true_coefficients[j]:10.5f} "
              + " ".join(f"{original[name][j]:12.5f}"
                         for name in models))

    print()
    print(separator)
    print("Train and Test Performance")
    print(separator)
    print()
    predictions = {}
    for name, coefficients in models.items():
        train = regression_metrics(y_train, predict(z_train, coefficients))
        predictions[name] = predict(z_test, coefficients)
        test = regression_metrics(y_test, predictions[name])
        print(f"{name:12s} Train RMSE={train['rmse']:.6f} "
              f"Test RMSE={test['rmse']:.6f} "
              f"Test R2={test['r_squared']:.6f}")

    print()
    print(separator)
    print("Sparsity and Objective Function")
    print(separator)
    print()
    for method in methods:
        coefficients = models[method]
        ratio = 1.0 if method == "Lasso" else (
            0.0 if method == "Ridge" else l1_ratio
        )
        value = objective_function(
            z_train, y_train, coefficients[0], coefficients[1:],
            selected[method], ratio
        )
        zeros = np.count_nonzero(np.isclose(coefficients[1:], 0, atol=1e-8))
        print(f"{method:12s} Objective={value:.6f} "
              f"Zero slopes={zeros}/{x.shape[1]}")

    plot_cross_validation(penalties, cv_results, selected)
    plot_regularisation_paths(penalties, paths)
    plot_predictions(y_test, predictions)

if __name__ == "__main__":
    main()
