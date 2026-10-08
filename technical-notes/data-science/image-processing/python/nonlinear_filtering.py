"""
Comparison of linear and non-linear smoothing filters.

This script compares Gaussian, median, and bilateral filtering
for Gaussian noise and impulse noise.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def convolve(image, kernel):
    """
    Convolve an image with a two-dimensional kernel.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    kernel : numpy.ndarray
        Convolution kernel.

    Output
    ------
    numpy.ndarray
        Filtered image.
    """
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="reflect",
    )

    kernel = np.flip(
        kernel,
        axis=(0, 1),
    )

    output = np.zeros_like(
        image,
        dtype=float,
    )

    for i in range(height):
        for j in range(width):
            neighborhood = padded[
                i:i + kernel_height,
                j:j + kernel_width,
            ]

            output[i, j] = np.sum(
                neighborhood * kernel
            )

    return output


def gaussian_kernel(size, sigma):
    """
    Construct a two-dimensional Gaussian kernel.

    Input
    -----
    size : int
        Width and height of the kernel.
    sigma : float
        Gaussian scale parameter.

    Output
    ------
    numpy.ndarray
        Normalized Gaussian kernel.
    """
    radius = size // 2

    coordinates = np.arange(
        -radius,
        radius + 1,
    )

    x, y = np.meshgrid(
        coordinates,
        coordinates,
    )

    kernel = np.exp(
        -(x**2 + y**2)
        / (2 * sigma**2)
    )

    kernel /= np.sum(kernel)

    return kernel


def median_filter(image, size):
    """
    Apply a median filter to an image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    size : int
        Width and height of the neighborhood.

    Output
    ------
    numpy.ndarray
        Median-filtered image.
    """
    radius = size // 2

    padded = np.pad(
        image,
        radius,
        mode="reflect",
    )

    output = np.zeros_like(
        image,
        dtype=float,
    )

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            neighborhood = padded[
                i:i + size,
                j:j + size,
            ]

            output[i, j] = np.median(
                neighborhood
            )

    return output


def bilateral_filter(
    image,
    size,
    sigma_s,
    sigma_r,
):
    """
    Apply a bilateral filter to an image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    size : int
        Width and height of the neighborhood.
    sigma_s : float
        Spatial scale parameter.
    sigma_r : float
        Range scale parameter.

    Output
    ------
    numpy.ndarray
        Bilaterally filtered image.
    """
    radius = size // 2

    coordinates = np.arange(
        -radius,
        radius + 1,
    )

    x, y = np.meshgrid(
        coordinates,
        coordinates,
    )

    spatial_weights = np.exp(
        -(x**2 + y**2)
        / (2 * sigma_s**2)
    )

    padded = np.pad(
        image,
        radius,
        mode="reflect",
    )

    output = np.zeros_like(
        image,
        dtype=float,
    )

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            neighborhood = padded[
                i:i + size,
                j:j + size,
            ]

            center = image[i, j]

            range_weights = np.exp(
                -(neighborhood - center) ** 2
                / (2 * sigma_r**2)
            )

            weights = (
                spatial_weights
                * range_weights
            )

            output[i, j] = np.sum(
                weights * neighborhood
            ) / np.sum(weights)

    return output


def create_image(size):
    """
    Create a synthetic grayscale image.

    Input
    -----
    size : int
        Width and height of the image.

    Output
    ------
    numpy.ndarray
        Synthetic grayscale image.
    """
    image = np.full(
        (size, size),
        0.2,
        dtype=float,
    )

    image[
        size // 4:3 * size // 4,
        size // 4:3 * size // 4,
    ] = 0.8

    return image


def add_gaussian_noise(image, sigma, rng):
    """
    Add Gaussian noise to an image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    sigma : float
        Noise standard deviation.
    rng : numpy.random.Generator
        Random number generator.

    Output
    ------
    numpy.ndarray
        Noisy image.
    """
    noise = rng.normal(
        0,
        sigma,
        image.shape,
    )

    return np.clip(
        image + noise,
        0,
        1,
    )


def add_impulse_noise(image, probability, rng):
    """
    Add impulse noise to an image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    probability : float
        Probability of pixel corruption.
    rng : numpy.random.Generator
        Random number generator.

    Output
    ------
    numpy.ndarray
        Noisy image.
    """
    output = image.copy()

    random_values = rng.random(
        image.shape
    )

    output[
        random_values < probability / 2
    ] = 0

    output[
        random_values > 1 - probability / 2
    ] = 1

    return output


def main():
    """
    Compare smoothing filters for two types of noise.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated comparison figure.
    """
    rng = np.random.default_rng(42)

    image = create_image(
        size=128
    )

    gaussian_noise = add_gaussian_noise(
        image,
        sigma=0.10,
        rng=rng,
    )

    impulse_noise = add_impulse_noise(
        image,
        probability=0.08,
        rng=rng,
    )

    gaussian_kernel_2d = gaussian_kernel(
        size=5,
        sigma=1.0,
    )

    rows = []

    for noisy_image in [
        gaussian_noise,
        impulse_noise,
    ]:
        gaussian_result = convolve(
            noisy_image,
            gaussian_kernel_2d,
        )

        median_result = median_filter(
            noisy_image,
            size=5,
        )

        bilateral_result = bilateral_filter(
            noisy_image,
            size=5,
            sigma_s=1.5,
            sigma_r=0.15,
        )

        rows.append([
            noisy_image,
            gaussian_result,
            median_result,
            bilateral_result,
        ])

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6),
    )

    titles = [
        "Noisy image",
        "Gaussian",
        "Median",
        "Bilateral",
    ]

    row_labels = [
        "Gaussian noise",
        "Impulse noise",
    ]

    for i in range(2):
        for j in range(4):
            axes[i, j].imshow(
                rows[i][j],
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[i, j].axis("off")

            if i == 0:
                axes[i, j].set_title(
                    titles[j]
                )

        axes[i, 0].set_ylabel(
            row_labels[i]
        )

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "nonlinear-filtering.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()