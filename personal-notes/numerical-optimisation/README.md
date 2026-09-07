# Numerical Optimisation

**Technical Notes**
**Author:** Rodrigo Kang

This directory contains systematic notes on numerical optimisation, with an emphasis on the mathematical foundations and computational methods used to formulate and solve optimisation problems.

The notes develop optimisation methods from their theoretical foundations through to their numerical implementation, connecting the geometry and analytical properties of optimisation problems with the behaviour of computational algorithms. Numerical optimisation provides a fundamental set of tools across applied mathematics, statistics, machine learning, operations research, engineering, and other quantitative disciplines, particularly when analytical solutions are unavailable or impractical.

## Scope

The material covers topics such as unconstrained and constrained optimisation, convexity, first- and second-order methods, line-search and trust-region methods, nonlinear least squares, stochastic optimisation, and numerical methods for large-scale problems.

Particular attention is given to optimality conditions, convergence properties, numerical stability, computational efficiency, and the structural properties of objective functions and constraints that determine which optimisation methods are appropriate.

The notes are organised into two broad areas:

* **[Unconstrained Optimisation](./unconstrained-optimisation/)** — Problems in which the optimisation variables are not subject to explicit constraints. This includes first- and second-order methods such as Gradient Descent, Newton's method, and related numerical approaches.

* **[Constrained Optimisation](./constrained-optimisation/)** — Problems in which the solution must satisfy equality, inequality, or other explicit constraints. This includes linear and nonlinear constrained optimisation and the methods used to characterise and compute their solutions.

This division provides the main organisational structure of the notes rather than an exhaustive classification of optimisation problems. Other important properties, including linearity, convexity, continuity, stochasticity, and problem scale, are considered within these areas where relevant.

## Approach

The notes are developed from first principles where this helps clarify the mathematical structure of an optimisation problem or the reasoning behind an algorithm. Analytical derivations are complemented by numerical implementations and computational experiments where appropriate.

The emphasis is not only on finding numerical solutions, but also on understanding:

* how optimisation problems are formulated;
* the mathematical properties of objective functions and constraints;
* how optimality conditions are derived and interpreted;
* how optimisation algorithms are constructed;
* the convergence properties of different methods;
* the role of numerical accuracy, stability, and computational cost;
* how problem structure influences the choice of algorithm; and
* where particular optimisation methods become ineffective or inappropriate.

