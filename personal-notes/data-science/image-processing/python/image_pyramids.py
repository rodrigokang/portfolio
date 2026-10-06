"""
Gaussian and Laplacian image pyramids.

This script constructs Gaussian and Laplacian pyramids
and reconstructs the original image from the Laplacian pyramid.
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
    kernel = np.flip(
        kernel,
        axis=(0, 1),
    )

    kernel_height, kernel_width = (
        kernel.shape
    )

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        image,
        (
            (pad_y, pad_y),
            (pad_x, pad_x),
        ),
        mode="reflect",
    )

    output = np.zeros_like(
        image,
        dtype=float,
    )

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[
                i:i + kernel_height,
                j:j + kernel_width,
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output


def binomial_kernel():
    """
    Create a two-dimensional binomial smoothing kernel.

    Input
    -----
    None

    Output
    ------
    numpy.ndarray
        Normalized two-dimensional smoothing kernel.
    """
    kernel_1d = np.array(
        [1, 4, 6, 4, 1],
        dtype=float,
    )

    kernel_1d /= np.sum(
        kernel_1d
    )

    kernel = np.outer(
        kernel_1d,
        kernel_1d,
    )

    return kernel


def reduce_image(image, kernel):
    """
    Smooth and downsample an image by a factor of two.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    kernel : numpy.ndarray
        Low-pass filter.

    Output
    ------
    numpy.ndarray
        Reduced image.
    """
    smoothed = convolve(
        image,
        kernel,
    )

    return smoothed[
        ::2,
        ::2,
    ]


def expand_image(image, target_shape):
    """
    Upsample an image using bilinear interpolation.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    target_shape : tuple
        Shape of the expanded image.

    Output
    ------
    numpy.ndarray
        Expanded image.
    """
    target_height, target_width = (
        target_shape
    )

    source_height, source_width = (
        image.shape
    )

    y_coordinates = np.linspace(
        0,
        source_height - 1,
        target_height,
    )

    x_coordinates = np.linspace(
        0,
        source_width - 1,
        target_width,
    )

    output = np.zeros(
        target_shape,
        dtype=float,
    )

    for i, y in enumerate(y_coordinates):
        y0 = int(np.floor(y))
        y1 = min(
            y0 + 1,
            source_height - 1,
        )

        beta = y - y0

        for j, x in enumerate(x_coordinates):
            x0 = int(np.floor(x))
            x1 = min(
                x0 + 1,
                source_width - 1,
            )

            alpha = x - x0

            output[i, j] = (
                (1 - alpha)
                * (1 - beta)
                * image[y0, x0]
                + alpha
                * (1 - beta)
                * image[y0, x1]
                + (1 - alpha)
                * beta
                * image[y1, x0]
                + alpha
                * beta
                * image[y1, x1]
            )

    return output


def gaussian_pyramid(image, levels):
    """
    Construct a Gaussian pyramid.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    levels : int
        Number of pyramid levels.

    Output
    ------
    list
        Gaussian pyramid levels.
    """
    kernel = binomial_kernel()

    pyramid = [
        image.astype(float)
    ]

    for _ in range(1, levels):
        reduced = reduce_image(
            pyramid[-1],
            kernel,
        )

        pyramid.append(
            reduced
        )

    return pyramid


def laplacian_pyramid(gaussian):
    """
    Construct a Laplacian pyramid.

    Input
    -----
    gaussian : list
        Gaussian pyramid levels.

    Output
    ------
    list
        Laplacian pyramid levels.
    """
    pyramid = []

    for level in range(
        len(gaussian) - 1
    ):
        expanded = expand_image(
            gaussian[level + 1],
            gaussian[level].shape,
        )

        laplacian = (
            gaussian[level]
            - expanded
        )

        pyramid.append(
            laplacian
        )

    pyramid.append(
        gaussian[-1]
    )

    return pyramid


def reconstruct(laplacian):
    """
    Reconstruct an image from a Laplacian pyramid.

    Input
    -----
    laplacian : list
        Laplacian pyramid levels.

    Output
    ------
    numpy.ndarray
        Reconstructed image.
    """
    image = laplacian[-1]

    for level in range(
        len(laplacian) - 2,
        -1,
        -1,
    ):
        image = expand_image(
            image,
            laplacian[level].shape,
        )

        image = (
            image
            + laplacian[level]
        )

    return image


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
        Synthetic image.
    """
    y, x = np.ogrid[
        :size,
        :size,
    ]

    image = np.full(
        (size, size),
        0.15,
        dtype=float,
    )

    image[
        24:104,
        24:104,
    ] = 0.45

    image[
        48:80,
        48:80,
    ] = 0.85

    circle = (
        (x - 94) ** 2
        + (y - 94) ** 2
        <= 14 ** 2
    )

    image[circle] = 1.0

    return image


def plot_pyramids(
    gaussian,
    laplacian,
    reconstruction,
    output_path,
):
    """
    Plot the Gaussian and Laplacian pyramids.

    Input
    -----
    gaussian : list
        Gaussian pyramid levels.
    laplacian : list
        Laplacian pyramid levels.
    reconstruction : numpy.ndarray
        Reconstructed image.
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    levels = len(
        gaussian
    )

    figure, axes = plt.subplots(
        2,
        levels,
        figsize=(12, 6),
    )

    for level in range(levels):
        axes[0, level].imshow(
            gaussian[level],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[0, level].set_title(
            rf"$G_{level}$"
        )

        if level < levels - 1:
            maximum = np.max(
                np.abs(
                    laplacian[level]
                )
            )

            axes[1, level].imshow(
                laplacian[level],
                cmap="gray",
                vmin=-maximum,
                vmax=maximum,
            )

            axes[1, level].set_title(
                rf"$L_{level}$"
            )

        else:
            axes[1, level].imshow(
                reconstruction,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[1, level].set_title(
                "Reconstruction"
            )

    axes[0, 0].set_ylabel(
        "Gaussian"
    )

    axes[1, 0].set_ylabel(
        "Laplacian"
    )

    for axis in axes.flat:
        axis.axis("off")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main():
    """
    Run the image pyramid experiment.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figure.
    """
    image = create_image(
        size=128
    )

    gaussian = gaussian_pyramid(
        image,
        levels=4,
    )

    laplacian = laplacian_pyramid(
        gaussian
    )

    reconstruction = reconstruct(
        laplacian
    )

    error = np.max(
        np.abs(
            image
            - reconstruction
        )
    )

    print(
        "Maximum reconstruction error:",
        error,
    )

    output_directory = (
        Path(__file__).resolve().parent.parent
        / "figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_pyramids(
        gaussian,
        laplacian,
        reconstruction,
        output_directory
        / "image-pyramids.png",
    )


if __name__ == "__main__":
    main()