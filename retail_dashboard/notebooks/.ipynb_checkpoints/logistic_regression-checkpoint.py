import numpy as np

def logistic_regression_log_likelihood(X, y, theta):
    """
    Compute the log-likelihood for logistic regression.

    Parameters:
    X : numpy array, shape (N, 2)
        Feature matrix where each row represents a sample and each column represents a feature.
        The first column should be filled with ones for the intercept term.
    y : numpy array, shape (N,)
        Target variable vector where each element represents the class label (0 or 1).
    theta : numpy array, shape (2,)
        Parameter vector for logistic regression. theta[0] is the intercept, and theta[1] is the coefficient for the predictor.

    Returns:
    log_likelihood : float
        Log-likelihood value.
    """
    N = len(y)  # Number of samples
    z = X@theta  # Compute linear combination of features and parameters
    prob = 1 / (1 + np.exp(-z))  # Calculate probability of class 1
    log_likelihood = -np.sum(np.log(1 + np.exp(z))) + np.sum(y * z)  # Compute log-likelihood

    return log_likelihood

import matplotlib.pyplot as plt

# Generar datos de ejemplo
N = 1000
X = np.column_stack((np.ones(N), np.random.randn(N)))
y = np.random.randint(2, size=N)

# Definir rango de valores para theta0 y theta1
theta0_range = np.linspace(-5, 5, 100)
theta1_range = np.linspace(-5, 5, 100)

# Calcular log-verosimilitud para cada valor de theta0 y theta1
log_likelihood_theta0 = np.zeros_like(theta0_range)
log_likelihood_theta1 = np.zeros_like(theta1_range)

for i in range(len(theta0_range)):
    theta = np.array([theta0_range[i], 0])  # Mantener theta1 constante en 0
    log_likelihood_theta0[i] = logistic_regression_log_likelihood(X, y, theta)

for i in range(len(theta1_range)):
    theta = np.array([0, theta1_range[i]])  # Mantener theta0 constante en 0
    log_likelihood_theta1[i] = logistic_regression_log_likelihood(X, y, theta)

# Crear gráficos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(theta0_range, log_likelihood_theta0)
plt.xlabel(r'$\theta_0$')
plt.ylabel('Log-Likelihood')
plt.title(r'Log-Likelihood vs. $\theta_0$')

plt.subplot(1, 2, 2)
plt.plot(theta1_range, log_likelihood_theta1)
plt.xlabel(r'$\theta_1$')
plt.ylabel('Log-Likelihood')
plt.title(r'Log-Likelihood vs. $\theta_1$')

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Generar datos de ejemplo
N = 1000
X = np.column_stack((np.ones(N), np.random.randn(N)))
y = np.random.randint(2, size=N)

# Definir rango de valores para theta0 y theta1
theta0_range = np.linspace(-5, 5, 100)
theta1_range = np.linspace(-5, 5, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)

# Calcular log-verosimilitud para cada combinación de theta0 y theta1
log_likelihood_grid = np.zeros_like(theta0_grid)
for i in range(len(theta0_range)):
    for j in range(len(theta1_range)):
        theta = np.array([theta0_range[i], theta1_range[j]])
        log_likelihood_grid[j, i] = logistic_regression_log_likelihood(X, y, theta)

# Crear gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_grid, theta1_grid, log_likelihood_grid, cmap='viridis')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel('Log-Likelihood')
ax.set_title('Log-Likelihood Surface')
plt.show()

def logistic_regression_log_likelihood_derivative(X, y, theta):
    """
    Compute the derivative of the log-likelihood function for logistic regression.

    Parameters:
    X : numpy array, shape (N, M)
        Feature matrix where each row represents a sample and each column represents a feature.
        The first column should be filled with ones for the intercept term.
    y : numpy array, shape (N,)
        Target variable vector where each element represents the class label (0 or 1).
    theta : numpy array, shape (M,)
        Parameter vector for logistic regression.

    Returns:
    derivative : numpy array, shape (M,)
        Derivative of the log-likelihood function with respect to each parameter.
    """
    z = X@theta
    prob = 1 / (1 + np.exp(-z))
    derivative = X.T@(y - prob)
    return derivative

def logistic_regression_log_likelihood_second_derivative(X, y, theta):
    """
    Compute the second derivative of the log-likelihood function for logistic regression.

    Parameters:
    X : numpy array, shape (N, M)
        Feature matrix where each row represents a sample and each column represents a feature.
        The first column should be filled with ones for the intercept term.
    y : numpy array, shape (N,)
        Target variable vector where each element represents the class label (0 or 1).
    theta : numpy array, shape (M,)
        Parameter vector for logistic regression.

    Returns:
    second_derivative : numpy array, shape (M, M)
        Second derivative of the log-likelihood function with respect to each parameter.
    """
    z = np.dot(X, theta)
    prob = 1 / (1 + np.exp(-z))
    W = np.diag(prob * (1 - prob))
    second_derivative = - X.T @ W @ X
    return second_derivative

def newton_raphson(X, y, initial_theta, max_iter, tol):
    """
    Apply the Newton-Raphson optimization method to find the optimal parameters for logistic regression.

    Parameters:
    X : numpy array, shape (N, M)
        Feature matrix where each row represents a sample and each column represents a feature.
        The first column should be filled with ones for the intercept term.
    y : numpy array, shape (N,)
        Target variable vector where each element represents the class label (0 or 1).
    initial_theta : numpy array, shape (M,)
        Initial guess for the parameter vector.
    max_iter : int, optional
        Maximum number of iterations for the optimization algorithm. Default is 100.
    tol : float, optional
        Tolerance for the convergence criterion. The optimization stops when the change in parameters
        is smaller than this value. Default is 1e-6.

    Returns:
    theta : numpy array, shape (M,)
        Optimal parameter vector found by the Newton-Raphson method.
    """
    theta = initial_theta
    iter_count = 0
    while iter_count < max_iter:
        f = logistic_regression_log_likelihood_derivative(X, y, theta)
        f_prime = logistic_regression_log_likelihood_second_derivative(X, y, theta)
        theta += -np.linalg.inv(f_prime) @ f
        
        # Check for convergence
        if np.linalg.norm(f) < tol:
            break
        
        iter_count += 1
    
    return theta

np.random.seed(12)

def generate_logistic_data(N, M, noise=0.1):
    """
    Generate synthetic logistic regression data.

    Parameters:
    N : int
        Number of samples.
    noise : float, optional
        Standard deviation of the Gaussian noise to be added to the data. Default is 0.1.

    Returns:
    X : numpy array, shape (N, 2)
        Feature matrix with an additional column for the intercept term.
    y : numpy array, shape (N,)
        Target variable vector.
    """
    # Generate feature matrix with intercept column
    X = np.random.randn(N, 1)
    intercept_column = np.ones((N, 1))
    X = np.hstack((intercept_column, X))

    # Generate parameter vector for generating data
    true_theta = np.random.randn(M)

    # Generate target variable based on logistic function
    z = X @ true_theta
    prob = 1 / (1 + np.exp(-z))
    y = np.random.binomial(1, prob)

    # Add Gaussian noise to the features
    X[:, 1] += np.random.normal(scale=noise, size=(N,))

    return X, y

# Generate example data
N = 1000
M = 2
X, y = generate_logistic_data(N, M)

# Initial value of theta
initial_theta = np.zeros(M)

# Number of iterations
n_iter = 1e6

# Tolerance
tol = 1e-6

# Apply the Newton-Raphson method
theta_optimal = newton_raphson(X, y, initial_theta, n_iter, tol)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 1], y, color='blue', label='Data')
plt.xlabel('X')
plt.ylabel('y')

# Plot the logistic function with optimal parameters
x_values = np.linspace(min(X[:, 1]), max(X[:, 1]), N)
logistic_function = 1 / (1 + np.exp(-(theta_optimal[0] + theta_optimal[1] * x_values)))
plt.plot(x_values, logistic_function, color='red', label='Logistic Function')

# Plot threshold line at 0.5
plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold (0.5)')

# Display the optimal parameters
plt.text(0.05, 0.9, fr'$\theta_0$ = {theta_optimal[0]:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', color='red')
plt.text(0.05, 0.85, fr'$\theta_1$ = {theta_optimal[1]:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', color='red')

plt.title('Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()