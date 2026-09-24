"""
Multiple linear regression: estimation, optimisation and inference.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f, norm, t


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Data and Design Matrix
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
    x2 = 0.5 * x1 + rng.normal(scale=np.sqrt(0.75), size=n)
    x3 = rng.normal(size=n)
    x = np.column_stack((x1, x2, x3))
    design = design_matrix(x)
    y = design @ coefficients + rng.normal(scale=noise_std, size=n)
    return x, y


def design_matrix(x):
    """
    Add an intercept column to the predictor matrix.

    Input:
        Predictor matrix of shape (N, M).

    Output:
        Design matrix of shape (N, M + 1).
    """
    return np.column_stack((np.ones(x.shape[0]), x))


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Least Squares and Optimality Conditions
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def objective_function(design, y, coefficients):
    """
    Evaluate the residual sum of squares.

    Input:
        Design matrix, response and coefficient vector.

    Output:
        Residual sum of squares.
    """
    residuals = y - design @ coefficients
    return float(residuals @ residuals)


def objective_gradient(design, y, coefficients):
    """
    Evaluate the gradient of the residual sum of squares.

    Input:
        Design matrix, response and coefficient vector.

    Output:
        Gradient vector.
    """
    return 2 * design.T @ (design @ coefficients - y)


def objective_hessian(design):
    """
    Evaluate the Hessian of the residual sum of squares.

    Input:
        Design matrix.

    Output:
        Hessian matrix.
    """
    return 2 * design.T @ design


def ordinary_least_squares(design, y):
    """
    Estimate coefficients by a least-squares factorisation.

    Input:
        Full-column-rank design matrix and response.

    Output:
        OLS coefficient vector.
    """
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("OLS inference requires a full-column-rank design matrix.")
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def bfgs_method(design, y, initial, tolerance=1e-8,
                max_iterations=100):
    """
    Minimise RSS by BFGS with exact line search.

    Input:
        Design matrix, response, initial vector and stopping settings.

    Output:
        Coefficients, iteration history and convergence flag.
    """
    coefficients = np.asarray(initial, dtype=float).copy()
    history = [coefficients.copy()]
    p = len(coefficients)
    inverse_hessian = np.eye(p)
    identity = np.eye(p)
    initial_norm = np.linalg.norm(objective_gradient(design, y, coefficients))
    threshold = tolerance * max(1.0, initial_norm)

    for _ in range(max_iterations):
        gradient = objective_gradient(design, y, coefficients)
        if np.linalg.norm(gradient) <= threshold:
            break

        direction = -inverse_hessian @ gradient
        projected = design @ direction
        curvature = 2 * (projected @ projected)
        if curvature <= 0:
            break
        step = -(gradient @ direction) / curvature

        updated = coefficients + step * direction
        new_gradient = objective_gradient(design, y, updated)
        displacement = updated - coefficients
        gradient_change = new_gradient - gradient
        curvature_pair = gradient_change @ displacement

        coefficients = updated
        history.append(coefficients.copy())
        if np.linalg.norm(new_gradient) <= threshold:
            break
        if curvature_pair <= 0:
            break

        rho = 1 / curvature_pair
        left = identity - rho * np.outer(displacement, gradient_change)
        right = identity - rho * np.outer(gradient_change, displacement)
        inverse_hessian = (
            left @ inverse_hessian @ right
            + rho * np.outer(displacement, displacement)
        )

    converged = np.linalg.norm(
        objective_gradient(design, y, coefficients)
    ) <= threshold
    return coefficients, np.asarray(history), converged


def newton_method(design, y, initial, tolerance=1e-8,
                  max_iterations=100):
    """
    Minimise RSS using Newton's method.

    Input:
        Design matrix, response, initial vector and stopping settings.

    Output:
        Coefficients, iteration history and convergence flag.
    """
    coefficients = np.asarray(initial, dtype=float).copy()
    history = [coefficients.copy()]
    hessian = objective_hessian(design)
    initial_norm = np.linalg.norm(objective_gradient(design, y, coefficients))
    threshold = tolerance * max(1.0, initial_norm)

    for _ in range(max_iterations):
        gradient = objective_gradient(design, y, coefficients)
        if np.linalg.norm(gradient) <= threshold:
            return coefficients, np.array(history), True
        coefficients -= np.linalg.solve(hessian, gradient)
        history.append(coefficients.copy())

    converged = np.linalg.norm(
        objective_gradient(design, y, coefficients)
    ) <= threshold
    return coefficients, np.array(history), converged


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Fit, ANOVA and Statistical Inference
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def regression_statistics(design, y, coefficients, alpha=0.05):
    """
    Calculate fit statistics and classical OLS inference.

    Input:
        Design matrix, response, OLS coefficients and alpha.

    Output:
        Dictionary of fitted values, fit statistics and inference.
    """
    n, p = design.shape
    if n <= p:
        raise ValueError("Residual degrees of freedom must be positive.")

    fitted = design @ coefficients
    residuals = y - fitted
    rss = float(residuals @ residuals)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    ess = tss - rss
    df_residual = n - p
    df_model = p - 1
    mse = rss / df_residual
    sigma_hat = np.sqrt(mse)
    r_squared = 1 - rss / tss
    adjusted_r_squared = 1 - (rss / df_residual) / (tss / (n - 1))

    gram_inverse = np.linalg.solve(design.T @ design, np.eye(p))
    covariance = mse * gram_inverse
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = coefficients / standard_errors
    p_values = 2 * t.sf(np.abs(t_values), df_residual)
    critical_t = t.ppf(1 - alpha / 2, df_residual)
    confidence_intervals = np.column_stack((
        coefficients - critical_t * standard_errors,
        coefficients + critical_t * standard_errors,
    ))
    f_statistic = (ess / df_model) / mse
    f_p_value = f.sf(f_statistic, df_model, df_residual)

    return {
        "fitted": fitted,
        "residuals": residuals,
        "rss": rss,
        "ess": ess,
        "tss": tss,
        "mse": mse,
        "sigma_hat": sigma_hat,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "df_model": df_model,
        "df_residual": df_residual,
        "covariance": covariance,
        "gram_inverse": gram_inverse,
        "standard_errors": standard_errors,
        "t_values": t_values,
        "p_values": p_values,
        "confidence_intervals": confidence_intervals,
        "critical_t": critical_t,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
    }


def prediction_intervals(new_design, coefficients, results):
    """
    Compute confidence and prediction intervals at new points.

    Input:
        New design matrix, coefficients and regression statistics.

    Output:
        Mean predictions, confidence bounds and prediction bounds.
    """
    mean = new_design @ coefficients
    quadratic = np.sum((new_design @ results["gram_inverse"])
                       * new_design, axis=1)
    mean_se = results["sigma_hat"] * np.sqrt(quadratic)
    prediction_se = results["sigma_hat"] * np.sqrt(1 + quadratic)
    critical = results["critical_t"]
    confidence = (mean - critical * mean_se,
                  mean + critical * mean_se)
    prediction = (mean - critical * prediction_se,
                  mean + critical * prediction_se)
    return mean, confidence, prediction


def regression_diagnostics(design, results):
    """
    Compute leverage, studentised residuals and Cook's distance.

    Input:
        Design matrix and regression statistics.

    Output:
        Leverage, internally studentised residuals and Cook's D.
    """
    n, p = design.shape
    leverage = np.sum((design @ results["gram_inverse"])
                      * design, axis=1)
    standardised = results["residuals"] / (
        results["sigma_hat"] * np.sqrt(1 - leverage)
    )
    cooks_distance = (
        standardised ** 2 * leverage / (p * (1 - leverage))
    )
    return leverage, standardised, cooks_distance


def variance_inflation_factors(x):
    """
    Calculate the VIF of each predictor using auxiliary OLS fits.

    Input:
        Predictor matrix without an intercept.

    Output:
        Variance inflation factors.
    """
    n, m = x.shape
    vif = np.empty(m)
    for j in range(m):
        response = x[:, j]
        others = np.delete(x, j, axis=1)
        auxiliary = design_matrix(others)
        coefficients = ordinary_least_squares(auxiliary, response)
        rss = objective_function(auxiliary, response, coefficients)
        tss = np.sum((response - np.mean(response)) ** 2)
        vif[j] = 1 / (rss / tss)
    return vif


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def plot_fit(y, results):
    """
    Display observed versus fitted responses.

    Input:
        Response and regression statistics.

    Output:
        Displays a scatter plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, results["fitted"], alpha=0.35, s=14)
    limits = (min(y.min(), results["fitted"].min()),
              max(y.max(), results["fitted"].max()))
    ax.plot(limits, limits, linestyle="--", label="Perfect fit")
    ax.set(xlabel="Observed response", ylabel="Fitted response",
           title="Observed vs Fitted")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_coefficient_intervals(coefficients, results):
    """
    Display coefficient estimates with confidence intervals.

    Input:
        Coefficient vector and regression statistics.

    Output:
        Displays an interval plot.
    """
    intervals = results["confidence_intervals"]
    positions = np.arange(len(coefficients))
    errors = np.vstack((coefficients - intervals[:, 0],
                        intervals[:, 1] - coefficients))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(positions, coefficients, yerr=errors, fmt="o", capsize=4)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(positions, [f"Intercept" if j == 0 else f"x{j}"
                              for j in positions])
    ax.set(ylabel="Coefficient estimate",
           title="Coefficient Confidence Intervals")
    fig.tight_layout()
    plt.show()


def plot_optimisation(history_bfgs, history_newton, design, y):
    """
    Display RSS across optimisation updates.

    Input:
        Both parameter histories, design matrix and response.

    Output:
        Displays the convergence plot.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for history, name in ((history_bfgs, "BFGS"),
                          (history_newton, "Newton")):
        values = [objective_function(design, y, vector)
                  for vector in history]
        ax.plot(np.arange(len(values)), values, marker="o",
                markersize=3, label=name)
    ax.set(xlabel="Update", ylabel="RSS", title="Numerical Optimisation")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_diagnostics(results, standardised, cooks_distance):
    """
    Display residual, normal Q-Q and influence diagnostics.

    Input:
        Regression statistics, studentised residuals and Cook's D.

    Output:
        Displays three diagnostic figures.
    """
    residuals = results["residuals"]
    fitted = results["fitted"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(fitted, residuals, alpha=0.35, s=14)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Fitted response", ylabel="Residual",
           title="Residuals vs Fitted")
    fig.tight_layout()
    plt.show()

    ordered = np.sort(standardised)
    probabilities = (np.arange(len(ordered)) + 0.5) / len(ordered)
    theoretical = norm.ppf(probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(theoretical, ordered, alpha=0.45, s=14)
    bounds = (min(theoretical.min(), ordered.min()),
              max(theoretical.max(), ordered.max()))
    ax.plot(bounds, bounds, linestyle="--", linewidth=1)
    ax.set(xlabel="Theoretical normal quantiles",
           ylabel="Internally studentised residual quantiles",
           title="Normal Q-Q Plot")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(np.arange(len(cooks_distance)), cooks_distance,
               alpha=0.45, s=14)
    ax.axhline(4 / len(cooks_distance), linestyle="--",
               label="4/N reference")
    ax.set(xlabel="Observation", ylabel="Cook's distance",
           title="Influence Diagnostics")
    ax.legend()
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def main():
    """
    Fit, optimise, analyse and diagnose multiple linear regression.

    Input:
        None.

    Output:
        Prints estimation results and displays figures.
    """
    rng = np.random.default_rng(seed=42)
    n = 1000
    true_coefficients = np.array([1.0, 2.0, -1.5, 0.75])
    noise_std = 1.0
    alpha = 0.05
    separator = "<>" * 36

    x, y = generate_data(n, true_coefficients, noise_std, rng)
    design = design_matrix(x)
    estimated = ordinary_least_squares(design, y)
    results = regression_statistics(design, y, estimated, alpha)

    initial = np.zeros(design.shape[1])
    bfgs_parameters, bfgs_history, bfgs_converged = bfgs_method(
        design, y, initial
    )
    newton_parameters, newton_history, newton_converged = (
        newton_method(design, y, initial)
    )
    gradient = objective_gradient(design, y, estimated)
    eigenvalues = np.linalg.eigvalsh(objective_hessian(design))
    leverage, standardised, cooks_distance = (
        regression_diagnostics(design, results)
    )
    vif = variance_inflation_factors(x)

    print(separator)
    print("Multiple Linear Regression")
    print(separator)
    print()
    for j, (true, estimate) in enumerate(zip(true_coefficients, estimated)):
        name = "Intercept" if j == 0 else f"x{j}"
        print(f"{name:10s} true={true:10.6f}  estimated={estimate:10.6f}")
    print(f"Minimum RSS:          {results['rss']:.6f}")

    print()
    print(separator)
    print("Optimality Conditions")
    print(separator)
    print()
    print(f"Gradient:             {gradient}")
    print(f"Hessian eigenvalues:  {eigenvalues}")
    print(f"Zero gradient:        {np.allclose(gradient, 0)}")
    print(f"Positive definite:    {np.all(eigenvalues > 0)}")
    print(f"Full column rank:     {np.linalg.matrix_rank(design) == design.shape[1]}")

    print()
    print(separator)
    print("Numerical Optimisation")
    print(separator)
    print()
    print(f"BFGS:                 {bfgs_parameters}")
    print(f"BFGS updates:         {len(bfgs_history) - 1}")
    print(f"BFGS converged:       {bfgs_converged}")
    print()
    print(f"Newton:               {newton_parameters}")
    print(f"Newton updates:       {len(newton_history) - 1}")
    print(f"Newton converged:     {newton_converged}")
    print()
    print(f"BFGS matches OLS:     {np.allclose(bfgs_parameters, estimated)}")
    print(f"Newton matches OLS:   {np.allclose(newton_parameters, estimated)}")

    print()
    print(separator)
    print("Fit and ANOVA")
    print(separator)
    print()
    for name in ("rss", "ess", "tss", "mse", "sigma_hat",
                 "r_squared", "adjusted_r_squared", "df_model",
                 "df_residual", "f_statistic", "f_p_value"):
        print(f"{name:20s} {results[name]:.6g}")
    print(f"RSS + ESS = TSS:      "
          f"{np.allclose(results['rss'] + results['ess'], results['tss'])}")

    print()
    print(separator)
    print("Coefficient Inference (H0: coefficient = 0)")
    print(separator)
    print()
    for j, estimate in enumerate(estimated):
        name = "Intercept" if j == 0 else f"x{j}"
        lower, upper = results["confidence_intervals"][j]
        print(f"{name:10s} estimate={estimate:.6f} "
              f"SE={results['standard_errors'][j]:.6f} "
              f"t={results['t_values'][j]:.4f} "
              f"p={results['p_values'][j]:.4g} "
              f"CI=({lower:.6f}, {upper:.6f})")

    x_new = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -1.0]])
    new_design = design_matrix(x_new)
    mean, confidence, prediction = prediction_intervals(
        new_design, estimated, results
    )
    print()
    print(separator)
    print("Mean Confidence and Prediction Intervals")
    print(separator)
    print()
    for i, row in enumerate(x_new):
        print(f"x={row}: mean={mean[i]:.4f}, "
              f"CI=({confidence[0][i]:.4f}, {confidence[1][i]:.4f}), "
              f"PI=({prediction[0][i]:.4f}, {prediction[1][i]:.4f})")

    print()
    print(separator)
    print("Diagnostics")
    print(separator)
    print()
    print(f"Sum of leverage:      {np.sum(leverage):.6f}")
    print(f"Expected leverage:    {design.shape[1]}")
    print(f"Maximum Cook's D:     {np.max(cooks_distance):.6f}")
    print(f"Points above 4/N:     {np.sum(cooks_distance > 4 / n)}")
    for j, value in enumerate(vif, start=1):
        print(f"VIF x{j}:              {value:.6f}")

    plot_fit(y, results)
    plot_coefficient_intervals(estimated, results)
    plot_optimisation(bfgs_history, newton_history, design, y)
    plot_diagnostics(results, standardised, cooks_distance)


if __name__ == "__main__":
    main()
