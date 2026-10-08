"""
Laplacian of Gaussian filtering and scale selection.

This script applies the Laplacian of Gaussian at different spatial
scales and uses the scale-normalized LoG to estimate the
characteristic scale of circular structures.
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


def log_kernel(size, sigma):
    """
    Construct a Laplacian of Gaussian kernel.

    Input
    -----
    size : int
        Width and height of the kernel.
    sigma : float
        Gaussian scale parameter.

    Output
    ------
    numpy.ndarray
        Laplacian of Gaussian kernel.
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

    radius_squared = x**2 + y**2

    gaussian = (
        1
        / (2 * np.pi * sigma**2)
        * np.exp(
            -radius_squared
            / (2 * sigma**2)
        )
    )

    kernel = (
        radius_squared / sigma**4
        - 2 / sigma**2
    ) * gaussian

    kernel -= np.mean(kernel)

    return kernel


def create_image(size, circles):
    """
    Create an image containing circular blobs.

    Input
    -----
    size : int
        Width and height of the image.
    circles : list
        Circle centers and radii.

    Output
    ------
    numpy.ndarray
        Synthetic grayscale image.
    """
    y, x = np.ogrid[
        :size,
        :size,
    ]

    image = np.zeros(
        (size, size),
        dtype=float,
    )

    for center_x, center_y, radius in circles:
        mask = (
            (x - center_x) ** 2
            + (y - center_y) ** 2
            <= radius**2
        )

        image[mask] = 1.0

    return image


def kernel_size(sigma):
    """
    Determine the kernel size for a Gaussian scale.

    Input
    -----
    sigma : float
        Gaussian scale parameter.

    Output
    ------
    int
        Odd kernel size.
    """
    size = int(
        6 * sigma + 1
    )

    if size % 2 == 0:
        size += 1

    return size


def plot_scale_responses(image, sigmas, output_path):
    """
    Plot LoG responses at different spatial scales.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    sigmas : list
        Gaussian scale parameters.
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    responses = []

    for sigma in sigmas:
        size = kernel_size(
            sigma
        )

        kernel = log_kernel(
            size=size,
            sigma=sigma,
        )

        response = convolve(
            image,
            kernel,
        )

        responses.append(
            response
        )

    figure, axes = plt.subplots(
        1,
        len(sigmas) + 1,
        figsize=(15, 3),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis("off")

    for axis, response, sigma in zip(
        axes[1:],
        responses,
        sigmas,
    ):
        limit = np.max(
            np.abs(response)
        )

        axis.imshow(
            response,
            cmap="gray",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(
            rf"$\sigma={sigma}$"
        )

        axis.axis("off")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_scale_selection(image, circles, sigmas, output_path):
    """
    Plot scale-normalized LoG responses.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    circles : list
        Circle centers and radii.
    sigmas : numpy.ndarray
        Gaussian scale parameters.
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    responses = np.zeros(
        (len(circles), len(sigmas))
    )

    for j, sigma in enumerate(sigmas):
        size = kernel_size(
            sigma
        )

        kernel = log_kernel(
            size=size,
            sigma=sigma,
        )

        normalized_kernel = (
            sigma**2 * kernel
        )

        response = convolve(
            image,
            normalized_kernel,
        )

        for i, (center_x, center_y, _) in enumerate(circles):
            responses[i, j] = np.abs(
                response[
                    center_y,
                    center_x,
                ]
            )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    for i, (_, _, radius) in enumerate(circles):
        axis.plot(
            sigmas,
            responses[i],
            label=rf"$r={radius}$",
        )

        estimated_sigma = sigmas[
            np.argmax(responses[i])
        ]

        axis.axvline(
            estimated_sigma,
            linestyle="--",
            alpha=0.4,
        )

    axis.set_xlabel(
        r"Scale $\sigma$"
    )

    axis.set_ylabel(
        "Absolute normalized LoG response"
    )

    axis.set_title(
        "Scale selection with normalized LoG"
    )

    axis.legend()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    """
    Run the LoG experiments.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figures.
    """
    circles = [
        (55, 55, 6),
        (180, 55, 12),
        (65, 180, 20),
        (180, 180, 32),
    ]

    image = create_image(
        size=256,
        circles=circles,
    )

    output_directory = (
        Path(__file__).resolve().parent.parent
        / "figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_scale_responses(
        image=image,
        sigmas=[
            1.0,
            2.0,
            4.0,
            8.0,
        ],
        output_path=(
            output_directory
            / "log-scales.png"
        ),
    )

    plot_scale_selection(
        image=image,
        circles=circles,
        sigmas=np.linspace(
            1.0,
            26.0,
            51,
        ),
        output_path=(
            output_directory
            / "log-scale-selection.png"
        ),
    )


if __name__ == "__main__":
    main()