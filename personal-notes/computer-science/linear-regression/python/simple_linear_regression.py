"""
Simple Linear Regression — Computational Implementation.

NumPy: explicit estimators and numerical methods.
SciPy: distribution quantiles and survival functions only.
Matplotlib: figures displayed, not saved.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Synthetic Data
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def generate_data(n, intercept, slope, noise_std, rng):
    """
    Generate synthetic simple linear regression observations.

    Input:
        n: Number of observations.
        intercept: True intercept.
        slope: True slope.
        noise_std: Standard deviation of the errors.
        rng: NumPy random number generator.

    Output:
        x: Predictor values.
        y: Response values.
    """
    x = rng.normal(0, 1, n)
    errors = rng.normal(0, noise_std, n)
    y = intercept + slope * x + errors
    return x, y


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Objective Function
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def objective_function(x, y, intercept, slope):
    """
    Compute the residual sum of squares.

    Input:
        x: Predictor values.
        y: Response values.
        intercept: Regression intercept.
        slope: Regression slope.

    Output:
        Residual sum of squares.
    """
    residuals = y - (intercept + slope * x)
    return np.sum(residuals**2)


def objective_gradient(x, y, intercept, slope):
    """
    Compute the gradient of the residual sum of squares.

    Input:
        x: Predictor values.
        y: Response values.
        intercept: Regression intercept.
        slope: Regression slope.

    Output:
        Gradient ordered as intercept, slope.
    """
    residuals = y - (intercept + slope * x)
    return np.array([
        -2 * np.sum(residuals),
        -2 * np.sum(x * residuals),
    ])


def objective_hessian(x):
    """
    Compute the constant Hessian of the residual sum of squares.

    Input:
        x: Predictor values.

    Output:
        Hessian matrix.
    """
    n = len(x)
    return 2 * np.array([
        [n, np.sum(x)],
        [np.sum(x), np.sum(x**2)],
    ])


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Analytical Solution
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def ordinary_least_squares(x, y):
    """
    Compute analytical simple linear regression estimators.

    Input:
        x: Predictor values.
        y: Response values.

    Output:
        intercept: Estimated intercept.
        slope: Estimated slope.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.sum((x - x_mean)**2)
    if sxx <= 0:
        raise ValueError("Predictor values must vary.")
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    return intercept, slope


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Numerical Optimisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def bfgs_method(x, y, initial, tolerance=1e-9,
                max_iterations=100):
    """
    Minimise RSS by BFGS with exact line search.

    Input:
        x: Predictor values.
        y: Response values.
        initial: Initial intercept and slope.
        tolerance: Gradient norm tolerance.
        max_iterations: Maximum number of updates.

    Output:
        parameters: Final intercept and slope.
        history: Parameter values, including the initial point.
        converged: Whether the gradient tolerance was reached.
    """
    parameters = np.asarray(initial, dtype=float).copy()
    history = [parameters.copy()]
    inverse_hessian = np.eye(2)
    identity = np.eye(2)

    for _ in range(max_iterations):
        gradient = objective_gradient(x, y, *parameters)
        if np.linalg.norm(gradient) <= tolerance:
            break

        direction = -inverse_hessian @ gradient
        # Exact line search along the BFGS direction for quadratic RSS.
        x_direction = direction[0] + direction[1] * x
        curvature = 2 * np.dot(x_direction, x_direction)
        if curvature <= 0:
            break
        step = -np.dot(gradient, direction) / curvature

        new_parameters = parameters + step * direction
        new_gradient = objective_gradient(x, y, *new_parameters)
        displacement = new_parameters - parameters
        gradient_change = new_gradient - gradient
        curvature_pair = np.dot(gradient_change, displacement)

        parameters = new_parameters
        history.append(parameters.copy())

        if np.linalg.norm(new_gradient) <= tolerance:
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

    converged = (
        np.linalg.norm(objective_gradient(x, y, *parameters))
        <= tolerance
    )
    return parameters, np.asarray(history), converged


def newton_method(x, y, initial, tolerance=1e-9,
                  max_iterations=20):
    """
    Minimise RSS with full Newton steps.

    Input:
        x: Predictor values.
        y: Response values.
        initial: Initial intercept and slope.
        tolerance: Gradient norm tolerance.
        max_iterations: Maximum number of updates.

    Output:
        parameters: Final intercept and slope.
        history: Parameter values, including the initial point.
        converged: Whether the gradient tolerance was reached.
    """
    parameters = np.asarray(initial, dtype=float).copy()
    history = [parameters.copy()]
    hessian = objective_hessian(x)

    for _ in range(max_iterations):
        gradient = objective_gradient(x, y, *parameters)
        if np.linalg.norm(gradient) <= tolerance:
            break
        direction = np.linalg.solve(hessian, gradient)
        parameters -= direction
        history.append(parameters.copy())

    converged = (
        np.linalg.norm(objective_gradient(x, y, *parameters))
        <= tolerance
    )
    return parameters, np.asarray(history), converged


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Statistical Inference
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def regression_statistics(x, y, intercept, slope, alpha=0.05):
    """
    Compute fit statistics, coefficient inference and ANOVA.

    Input:
        x: Predictor values.
        y: Response values.
        intercept: Estimated intercept.
        slope: Estimated slope.
        alpha: Significance level.

    Output:
        results: Dictionary of fitted values and statistics.
    """
    n = len(x)
    if n <= 2:
        raise ValueError("At least three observations are required.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.sum((x - x_mean)**2)
    fitted = intercept + slope * x
    residuals = y - fitted

    rss = np.sum(residuals**2)
    tss = np.sum((y - y_mean)**2)
    ess = np.sum((fitted - y_mean)**2)
    df_error = n - 2
    mse = rss / df_error
    sigma_hat = np.sqrt(mse)

    se_intercept = np.sqrt(
        mse * (1 / n + x_mean**2 / sxx)
    )
    se_slope = np.sqrt(mse / sxx)
    estimates = np.array([intercept, slope])
    standard_errors = np.array([se_intercept, se_slope])

    t_values = estimates / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), df_error)
    t_critical = stats.t.ppf(1 - alpha / 2, df_error)
    confidence_intervals = np.column_stack((
        estimates - t_critical * standard_errors,
        estimates + t_critical * standard_errors,
    ))

    r_squared = 1 - rss / tss if tss > 0 else np.nan
    adjusted_r_squared = (
        1 - (rss / df_error) / (tss / (n - 1))
        if tss > 0 else np.nan
    )
    f_statistic = ess / mse
    f_p_value = stats.f.sf(f_statistic, 1, df_error)

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
        "standard_errors": standard_errors,
        "t_values": t_values,
        "p_values": p_values,
        "confidence_intervals": confidence_intervals,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "df_regression": 1,
        "df_error": df_error,
        "df_total": n - 1,
        "x_mean": x_mean,
        "sxx": sxx,
        "t_critical": t_critical,
    }


def prediction_intervals(x_new, intercept, slope, results):
    """
    Compute confidence and prediction intervals at new x values.

    Input:
        x_new: Predictor values for inference.
        intercept: Estimated intercept.
        slope: Estimated slope.
        results: Output from regression_statistics.

    Output:
        mean: Estimated conditional mean.
        confidence: Confidence interval for the mean.
        prediction: Prediction interval for a new observation.
    """
    x_new = np.asarray(x_new)
    n = results["df_total"] + 1
    variance_mean = results["mse"] * (
        1 / n
        + (x_new - results["x_mean"])**2 / results["sxx"]
    )
    mean = intercept + slope * x_new
    margin_mean = results["t_critical"] * np.sqrt(variance_mean)
    margin_prediction = results["t_critical"] * np.sqrt(
        results["mse"] + variance_mean
    )
    confidence = (mean - margin_mean, mean + margin_mean)
    prediction = (mean - margin_prediction,
                  mean + margin_prediction)
    return mean, confidence, prediction


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Diagnostics
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def regression_diagnostics(x, results):
    """
    Compute leverage, standardised residuals and Cook's 
    distance.

    Input:
        x: Predictor values.
        results: Output from regression_statistics.

    Output:
        leverage: Diagonal entries of the hat matrix.
        standardised: Internally studentised residuals.
        cooks_distance: Cook's distance for each observation.
    """
    n = len(x)
    leverage = (
        1 / n
        + (x - results["x_mean"])**2 / results["sxx"]
    )
    standardised = results["residuals"] / np.sqrt(
        results["mse"] * (1 - leverage)
    )
    cooks_distance = (
        standardised**2 * leverage / (2 * (1 - leverage))
    )
    return leverage, standardised, cooks_distance


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Visualisation
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def plot_data(x, y, intercept, slope, estimated_intercept,
              estimated_slope, results):
    """
    Plot data, true line, OLS line and inferential intervals.

    Input:
        x: Predictor values.
        y: Response values.
        intercept: True intercept.
        slope: True slope.
        estimated_intercept: Estimated intercept.
        estimated_slope: Estimated slope.
        results: Output from regression_statistics.

    Output:
        Displays the regression fit.
    """
    x_line = np.linspace(x.min(), x.max(), 200)
    y_true = intercept + slope * x_line
    mean, confidence, prediction = prediction_intervals(
        x_line, estimated_intercept, estimated_slope, results
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x, y, s=18, alpha=0.3, label="Observations")
    ax.fill_between(
        x_line, *prediction, alpha=0.12,
        label="95% prediction interval"
    )
    ax.fill_between(
        x_line, *confidence, alpha=0.25,
        label="95% mean confidence interval"
    )
    ax.plot(
        x_line, y_true, color="black", linewidth=1.6,
        label="True regression line"
    )
    ax.plot(
        x_line, mean, color="tab:red", linestyle="--",
        linewidth=1.8, label="OLS regression line"
    )
    ax.set(
        title="Simple Linear Regression",
        xlabel="Predictor (x)",
        ylabel="Response (y)",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_objective_function(x, y, intercept, slope,
                            estimated_intercept, estimated_slope,
                            bfgs_history, newton_history):
    """
    Plot the RSS surface, its minimum and optimisation paths.

    Input:
        x: Predictor values.
        y: Response values.
        intercept: True intercept.
        slope: True slope.
        estimated_intercept: Estimated intercept.
        estimated_slope: Estimated slope.
        bfgs_history: BFGS parameter history.
        newton_history: Newton parameter history.

    Output:
        Displays the objective function surface.
    """
    intercept_values = np.linspace(intercept - 2, intercept + 2, 80)
    slope_values = np.linspace(slope - 2, slope + 2, 80)
    intercept_grid, slope_grid = np.meshgrid(
        intercept_values, slope_values
    )
    objective_grid = np.empty_like(intercept_grid)

    for i in range(intercept_grid.shape[0]):
        for j in range(intercept_grid.shape[1]):
            objective_grid[i, j] = objective_function(
                x, y, intercept_grid[i, j], slope_grid[i, j]
            )

    minimum = objective_function(
        x, y, estimated_intercept, estimated_slope
    )
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        intercept_grid, slope_grid, objective_grid,
        cmap="viridis", alpha=0.65, linewidth=0,
        antialiased=True
    )

    # Show early steps separately from the tightly clustered tail.
    for history, label, color in (
        (bfgs_history, "BFGS", "tab:orange"),
        (newton_history, "Newton", "tab:blue"),
    ):
        shown = history[:min(len(history), 30)]
        heights = [
            objective_function(x, y, *parameters)
            for parameters in shown
        ]
        ax.plot(
            shown[:, 0], shown[:, 1], heights,
            color=color, marker="o", markersize=3,
            linewidth=1.5, label=label
        )

    ax.scatter(
        estimated_intercept, estimated_slope, minimum,
        color="red", s=65, depthshade=False,
        label="OLS minimum"
    )
    ax.set(
        title="Least Squares Objective Function",
        xlabel="Intercept",
        ylabel="Slope",
        zlabel="Residual Sum of Squares",
    )
    ax.view_init(elev=25, azim=-135)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_diagnostics(x, results, standardised, cooks_distance):
    """
    Plot residual patterns, normal quantiles and influence.

    Input:
        x: Predictor values.
        results: Output from regression_statistics.
        standardised: Internally studentised residuals.
        cooks_distance: Cook's distance.

    Output:
        Displays three diagnostic figures.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        results["fitted"], results["residuals"],
        s=20, alpha=0.5
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="Residuals vs Fitted Values",
        xlabel="Fitted value",
        ylabel="Residual",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()

    theoretical = stats.norm.ppf(
        (np.arange(1, len(x) + 1) - 0.5) / len(x)
    )
    observed = np.sort(standardised)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(theoretical, observed, s=20, alpha=0.55)
    ax.plot(
        [theoretical[0], theoretical[-1]],
        [theoretical[0], theoretical[-1]],
        color="black", linestyle="--", linewidth=1
    )
    ax.set(
        title="Normal Q–Q Plot",
        xlabel="Theoretical normal quantile",
        ylabel="Standardised residual quantile",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        np.arange(len(x)), cooks_distance,
        s=20, alpha=0.6
    )
    ax.axhline(
        4 / len(x), color="tab:red", linestyle="--",
        linewidth=1, label="4 / N reference"
    )
    ax.set(
        title="Cook's Distance",
        xlabel="Observation index",
        ylabel="Cook's distance",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def main():
    """
    Fit, optimise, analyse and diagnose simple linear
    regression.

    Input:
        None.

    Output:
        Prints estimation results and displays figures.
    """
    rng = np.random.default_rng(seed=42)

    n = 1000
    intercept = 1.0
    slope = 2.0
    noise_std = 1.0
    alpha = 0.05

    separator = "<>" * 36

    x, y = generate_data(n, intercept, slope, noise_std, rng)

    estimated_intercept, estimated_slope = ordinary_least_squares(
        x, y
    )

    results = regression_statistics(
        x, y, estimated_intercept, estimated_slope, alpha
    )

    initial = np.array([-0.5, 0.5])

    bfgs_parameters, bfgs_history, bfgs_converged = bfgs_method(
        x, y, initial
    )

    newton_parameters, newton_history, newton_converged = (
        newton_method(x, y, initial)
    )

    gradient = objective_gradient(
        x, y, estimated_intercept, estimated_slope
    )

    eigenvalues = np.linalg.eigvalsh(objective_hessian(x))

    leverage, standardised, cooks_distance = (
        regression_diagnostics(x, results)
    )

    # Main Results

    print(separator)
    print("Simple Linear Regression")
    print(separator)
    print()

    print(f"True intercept:       {intercept:.6f}")
    print(f"Estimated intercept:  {estimated_intercept:.6f}")
    print(f"True slope:           {slope:.6f}")
    print(f"Estimated slope:      {estimated_slope:.6f}")
    print(f"Minimum RSS:          {results['rss']:.6f}")

    # Optimality Conditions

    print()
    print(separator)
    print("Optimality Conditions")
    print(separator)
    print()

    print(f"Gradient:             {gradient}")
    print(f"Hessian eigenvalues:  {eigenvalues}")
    print(f"Zero gradient:        {np.allclose(gradient, 0)}")
    print(f"Positive definite:    {np.all(eigenvalues > 0)}")

    # Numerical Optimisation

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

    print(
        "BFGS matches OLS:     "
        f"{np.allclose(bfgs_parameters, [estimated_intercept, estimated_slope])}"
    )
    print(
        "Newton matches OLS:   "
        f"{np.allclose(newton_parameters, [estimated_intercept, estimated_slope])}"
    )

    # Fit and ANOVA

    print()
    print(separator)
    print("Fit and ANOVA")
    print(separator)
    print()

    for name in (
        "rss",
        "ess",
        "tss",
        "mse",
        "sigma_hat",
        "r_squared",
        "adjusted_r_squared",
        "f_statistic",
        "f_p_value",
    ):
        print(f"{name:20s} {results[name]:.6g}")

    print(
        "RSS + ESS = TSS:      "
        f"{np.allclose(results['rss'] + results['ess'], results['tss'])}"
    )

    # Coefficient Inference

    print()
    print(separator)
    print("Coefficient Inference (H0: coefficient = 0)")
    print(separator)
    print()

    for index, name in enumerate(("Intercept", "Slope")):
        lower, upper = results["confidence_intervals"][index]

        estimate = (
            estimated_intercept
            if index == 0
            else estimated_slope
        )

        print(
            f"{name:10s} "
            f"estimate={estimate:.6f} "
            f"SE={results['standard_errors'][index]:.6f} "
            f"t={results['t_values'][index]:.4f} "
            f"p={results['p_values'][index]:.4g} "
            f"CI=({lower:.6f}, {upper:.6f})"
        )

    # Confidence and Prediction Intervals

    x_new = np.array([0.0, 1.0])

    mean, confidence, prediction = prediction_intervals(
        x_new, estimated_intercept, estimated_slope, results
    )

    print()
    print(separator)
    print("Mean Confidence and Prediction Intervals")
    print(separator)
    print()

    for i, value in enumerate(x_new):
        print(
            f"x={value:.1f}: mean={mean[i]:.4f}, "
            f"CI=({confidence[0][i]:.4f}, {confidence[1][i]:.4f}), "
            f"PI=({prediction[0][i]:.4f}, {prediction[1][i]:.4f})"
        )

    # Diagnostics

    print()
    print(separator)
    print("Diagnostics")
    print(separator)
    print()

    print(f"Sum of leverage:      {np.sum(leverage):.6f}")
    print(f"Maximum Cook's D:     {np.max(cooks_distance):.6f}")
    print(f"Points above 4/N:     {np.sum(cooks_distance > 4 / n)}")

    # Visualisation

    plot_data(
        x, y, intercept, slope,
        estimated_intercept, estimated_slope, results
    )

    plot_objective_function(
        x, y, intercept, slope,
        estimated_intercept, estimated_slope,
        bfgs_history, newton_history
    )

    plot_diagnostics(
        x, results, standardised, cooks_distance
    )


if __name__ == "__main__":
    main()