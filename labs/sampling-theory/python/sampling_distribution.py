
"""
Compute the sampling distribution of the sample mean with and without
replacement.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import itertools as it
import numpy as np

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Population parameters
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Population
population = np.array([2, 3, 6, 8, 11])

# Population size
population_size = len(population)

# Population mean
population_mean = np.mean(population)

# Population variance
population_variance = np.var(population)

# Population standard deviation
population_std = np.std(population)

print("<>" * 36)
print("Population Parameters")
print("<>" * 36)

print("Population mean:", population_mean)
print("Population variance:", population_variance)
print("Population standard deviation:", population_std)

print("<>" * 36)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling with replacement
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Sample size
sample_size = 2

# All possible samples with replacement
swr = np.array(
    list(it.product(population, repeat=sample_size))
)

# Sample means
swr_sample_means = np.mean(swr, axis=1)

# Mean of the sampling distribution of means
swr_mean = np.mean(swr_sample_means)

# Variance of the sampling distribution of means
swr_variance = np.var(swr_sample_means)

# Standard deviation of the sampling distribution of means
swr_std = np.std(swr_sample_means)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling without replacement
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# All possible samples without replacement
swor = np.array(
    list(it.combinations(population, sample_size))
)

# Sample means
swor_sample_means = np.mean(swor, axis=1)

# Mean of the sampling distribution of means
swor_mean = np.mean(swor_sample_means)

# Variance of the sampling distribution of means
swor_variance = np.var(swor_sample_means)

# Standard deviation of the sampling distribution of means
swor_std = np.std(swor_sample_means)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling with replacement: results
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

print("\n" + "<>" * 36)
print("Sampling With Replacement")
print("<>" * 36)

print("\nAll possible samples:")
print(swr)

print("\nSample means:")
print(swr_sample_means)

print("\nSampling distribution of means:")
print("Mean:", swr_mean)
print("Variance:", swr_variance)
print("Standard deviation:", swr_std)

print("\n" + "<>" * 36)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling without replacement: results
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

print("\n" + "<>" * 36)
print("Sampling Without Replacement")
print("<>" * 36)

print("\nAll possible samples:")
print(swor)

print("\nSample means:")
print(swor_sample_means)

print("\nSampling distribution of means:")
print("Mean:", swor_mean)
print("Variance:", swor_variance)
print("Standard deviation:", swor_std)

print("\n" + "<>" * 36)