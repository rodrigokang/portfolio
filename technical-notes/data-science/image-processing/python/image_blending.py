"""
Multi-resolution image blending.

This script compares direct image compositing with
Laplacian pyramid blending.
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

    kernel_height, kernel_width = kernel.shape

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

    kernel_1d /= np.sum(kernel_1d)

    return np.outer(
        kernel_1d,
        kernel_1d,
    )


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
    target_height, target_width = target_shape

    source_height, source_width = image.shape

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
        pyramid.append(
            reduce_image(
                pyramid[-1],
                kernel,
            )
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

        pyramid.append(
            gaussian[level]
            - expanded
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
        image = (
            laplacian[level]
            + expand_image(
                image,
                laplacian[level].shape,
            )
        )

    return image


def create_images(size):
    """
    Create two synthetic images and a blending mask.

    Input
    -----
    size : int
        Width and height of the images.

    Output
    ------
    tuple
        Two source images and a binary mask.
    """
    y, x = np.indices(
        (size, size)
    )

    image_a = (
        0.25
        + 0.45 * x / (size - 1)
    )

    image_b = (
        0.75
        - 0.45 * x / (size - 1)
    )

    texture_a = (
        ((x // 4) % 2) * 0.15
    )

    texture_b = (
        ((y // 4) % 2) * 0.15
    )

    image_a = np.clip(
        image_a + texture_a,
        0,
        1,
    )

    image_b = np.clip(
        image_b + texture_b,
        0,
        1,
    )

    mask = np.zeros(
        (size, size),
        dtype=float,
    )

    mask[:, :size // 2] = 1.0

    return (
        image_a,
        image_b,
        mask,
    )


def pyramid_blend(
    image_a,
    image_b,
    mask,
    levels,
):
    """
    Blend two images using Laplacian pyramids.

    Input
    -----
    image_a : numpy.ndarray
        First source image.
    image_b : numpy.ndarray
        Second source image.
    mask : numpy.ndarray
        Binary blending mask.
    levels : int
        Number of pyramid levels.

    Output
    ------
    numpy.ndarray
        Multi-resolution blended image.
    """
    gaussian_a = gaussian_pyramid(
        image_a,
        levels,
    )

    gaussian_b = gaussian_pyramid(
        image_b,
        levels,
    )

    gaussian_mask = gaussian_pyramid(
        mask,
        levels,
    )

    laplacian_a = laplacian_pyramid(
        gaussian_a
    )

    laplacian_b = laplacian_pyramid(
        gaussian_b
    )

    blended_pyramid = []

    for level in range(levels):
        weight = gaussian_mask[level]

        blended_level = (
            weight
            * laplacian_a[level]
            + (1 - weight)
            * laplacian_b[level]
        )

        blended_pyramid.append(
            blended_level
        )

    return reconstruct(
        blended_pyramid
    )


def plot_results(
    image_a,
    image_b,
    direct,
    pyramid,
    output_path,
):
    """
    Plot the image blending experiment.

    Input
    -----
    image_a : numpy.ndarray
        First source image.
    image_b : numpy.ndarray
        Second source image.
    direct : numpy.ndarray
        Direct composite.
    pyramid : numpy.ndarray
        Pyramid blend.
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3),
    )

    images = [
        image_a,
        image_b,
        direct,
        pyramid,
    ]

    titles = [
        "Image A",
        "Image B",
        "Direct Blend",
        "Pyramid Blend",
    ]

    for axis, image, title in zip(
        axes,
        images,
        titles,
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axis.set_title(title)
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
    Run the image blending experiment.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figure.
    """
    image_a, image_b, mask = (
        create_images(
            size=128
        )
    )

    direct = (
        mask * image_a
        + (1 - mask) * image_b
    )

    pyramid = pyramid_blend(
        image_a,
        image_b,
        mask,
        levels=5,
    )

    output_directory = (
        Path(__file__).resolve().parent.parent
        / "figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_results(
        image_a,
        image_b,
        direct,
        pyramid,
        output_directory
        / "image-blending.png",
    )


if __name__ == "__main__":
    main()