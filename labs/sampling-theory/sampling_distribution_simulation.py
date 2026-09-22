
"""
Simulate the sampling distribution of the sample mean with and without
replacement from a normally distributed population.
"""

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Import libraries
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import numpy as np
import matplotlib.pyplot as plt


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Population parameters
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Population size
population_size = 3000

# Population mean
population_mean = 68.0

# Population standard deviation
population_std = 3.0

# Seed to reproduce the simulation
seed = 42

# Random number generator
rng = np.random.default_rng(seed=seed)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Generate the population
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

population = rng.normal(
    loc=population_mean,
    scale=population_std,
    size=population_size
)

# Actual parameters of the generated population
actual_population_mean = np.mean(population)

actual_population_variance = np.var(population)

actual_population_std = np.std(population)

print("<>" * 36)
print("Population Parameters")
print("<>" * 36)

print("Mean:", population_mean)
print("Standard deviation:", population_std)

print("\nGenerated population parameters:")
print("Mean:", actual_population_mean)
print("Variance:", actual_population_variance)
print("Standard deviation:", actual_population_std)

print("<>" * 36)

# Square-root choice
population_bins = int(np.sqrt(population_size))

# Histogram
plt.hist(population, bins=population_bins)
plt.title("Population Distribution")
plt.xlabel("Heights (inches)")
plt.ylabel("Frequency")
plt.show()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling parameters
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Number of samples
number_of_samples = 80

# Sample size
sample_size = 25


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling with replacement
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

swr = rng.choice(
    population,
    size=(number_of_samples, sample_size),
    replace=True
)

# Sampling distribution of means
swr_sample_means = np.mean(swr, axis=1)

# Square-root choice
swr_bins = int(np.sqrt(len(swr_sample_means)))

# Histogram of the sampling distribution of means
plt.hist(swr_sample_means, bins=swr_bins)
plt.title(
    "Sampling Distribution of the Mean with Replacement"
)
plt.xlabel(r"Sample mean ($\bar{X}$)")
plt.ylabel("Frequency")
plt.show()

# Mean of sampling distribution of means
swr_mean = np.mean(swr_sample_means)

# Variance of sampling distribution of means
swr_variance = np.var(swr_sample_means)

# STD of sampling distribution of means
swr_std = np.std(swr_sample_means)

# Theoretical standard error: original population
swr_theoretical_std = (
    population_std / np.sqrt(sample_size)
)

# Theoretical standard error: generated population
swr_actual_theoretical_std = (
    actual_population_std / np.sqrt(sample_size)
)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling with replacement: results
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

print("\n" + "<>" * 36)
print("Sampling With Replacement")
print("<>" * 36)

print("\nAll samples:")
print(swr)

print("\nSample matrix shape:", swr.shape)

print("\nSampling distribution of means:")
print("Mean:", swr_mean)
print("Variance:", swr_variance)
print("Standard deviation:", swr_std)

print("\nTheoretical standard error:")
print("Original population:", swr_theoretical_std)
print("Generated population:", swr_actual_theoretical_std)

print("\n" + "<>" * 36)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling without replacement
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

swor = np.array([
    rng.choice(
        population,
        size=sample_size,
        replace=False
    )
    for _ in range(number_of_samples)
])

# Sampling distribution of means
swor_sample_means = np.mean(swor, axis=1)

# Square-root choice
swor_bins = int(np.sqrt(len(swor_sample_means)))

# Histogram of the sampling distribution of means
plt.hist(swor_sample_means, bins=swor_bins)
plt.title(
    "Sampling Distribution of the Mean without Replacement"
)
plt.xlabel(r"Sample mean ($\bar{X}$)")
plt.ylabel("Frequency")
plt.show()

# Mean of sampling distribution of means
swor_mean = np.mean(swor_sample_means)

# Variance of sampling distribution of means
swor_variance = np.var(swor_sample_means)

# STD of sampling distribution of means
swor_std = np.std(swor_sample_means)

# Finite population correction
fpc = np.sqrt(
    (population_size - sample_size)
    / (population_size - 1)
)

# Theoretical standard error: original population
swor_theoretical_std = (
    population_std / np.sqrt(sample_size)
) * fpc

# Theoretical standard error: generated population
swor_actual_theoretical_std = (
    actual_population_std / np.sqrt(sample_size)
) * fpc


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Sampling without replacement: results
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

print("\n" + "<>" * 36)
print("Sampling Without Replacement")
print("<>" * 36)

print("\nAll samples:")
print(swor)

print("\nSample matrix shape:", swor.shape)

print("\nSampling distribution of means:")
print("Mean:", swor_mean)
print("Variance:", swor_variance)
print("Standard deviation:", swor_std)

print("\nFinite population correction:", fpc)

print("\nTheoretical standard error:")
print("Original population:", swor_theoretical_std)
print("Generated population:", swor_actual_theoretical_std)

print("\n" + "<>" * 36)