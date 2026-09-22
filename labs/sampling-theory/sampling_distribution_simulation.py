# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Population parameters
population_size = 3000
population_mean = 68.0
population_std = 3.0

# Seed to reproduce the simulation
seed = 42

# Random number generator
rng = np.random.default_rng(seed = seed)

# Generate the population
population = rng.normal(
    loc=population_mean,
    scale=population_std,
    size=population_size
)

print(population)

# Square-root choice
population_bins = int(np.sqrt(population_size))

# Histogram
plt.hist(population, bins=population_bins)
plt.title("Population Distribution")
plt.xlabel("Heights (inches)")
plt.ylabel("Frequency")
plt.show()

# Sampling parameters
number_of_samples = 80
sample_size = 25

# Sampling with replacement
swr = rng.choice(
    population,
    size=(number_of_samples, sample_size),
    replace=True
)

print(swr)
print(swr.shape)

# Sampling distribution of means
swr_sample_means = np.mean(swr, axis=1)

# Square-root choice
swr_bins = int(np.sqrt(len(swr_sample_means)))

# Histrogram of the sampling distribution of means
plt.hist(swr_sample_means, bins=swr_bins)
plt.title("Sampling Distribution of the Mean")
plt.xlabel(r"Sample mean ($\bar{X}$)")
plt.show()

# Mean of sampling distribution of means
swr_mean = np.mean(swr_sample_means)
# Variance of sampling distribution of means
swr_variance = np.var(swr_sample_means)
# STD of sampling distribution of means
swr_std = np.std(swr_sample_means)

print(swr_mean)
print(swr_variance)
print(swr_std)