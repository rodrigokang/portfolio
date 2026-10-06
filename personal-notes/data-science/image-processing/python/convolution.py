"""
Linear filtering using two-dimensional convolution.

This script implements convolution from first principles and compares
box filtering, Gaussian filtering, and the Sobel operator.
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


def main():
    """
    Compare several linear filters.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated comparison figure.
    """
    rng = np.random.default_rng(42)

    image = np.zeros(
        (256, 256),
        dtype=float,
    )

    image[48:208, 48:208] = 0.4
    image[96:160, 96:160] = 0.9

    noise = rng.normal(
        loc=0.0,
        scale=0.08,
        size=image.shape,
    )

    noisy = np.clip(
        image + noise,
        0,
        1,
    )

    box = (
        1 / 9
        * np.ones((3, 3))
    )

    gaussian = gaussian_kernel(
        size=9,
        sigma=1.5,
    )

    sobel_x = (
        1 / 8
        * np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])
    )

    sobel_y = (
        1 / 8
        * np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ])
    )

    box_filtered = convolve(
        noisy,
        box,
    )

    gaussian_filtered = convolve(
        noisy,
        gaussian,
    )

    gradient_x = convolve(
        gaussian_filtered,
        sobel_x,
    )

    gradient_y = convolve(
        gaussian_filtered,
        sobel_y,
    )

    gradient = np.sqrt(
        gradient_x**2
        + gradient_y**2
    )

    results = [
        noisy,
        box_filtered,
        gaussian_filtered,
        gradient,
    ]

    titles = [
        "Original",
        "Box filter",
        "Gaussian filter",
        "Gradient magnitude",
    ]

    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(12, 3),
    )

    for axis, result, title in zip(
        axes,
        results,
        titles,
    ):
        axis.imshow(
            result,
            cmap="gray",
        )

        axis.set_title(title)
        axis.axis("off")

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "convolution.png"
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