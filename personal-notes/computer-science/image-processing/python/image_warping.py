"""
Image warping using an affine transformation.

This script implements inverse warping and bilinear
interpolation from first principles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bilinear_interpolation(image, x, y):
    """
    Interpolate an image at continuous coordinates.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    x, y : float
        Continuous image coordinates.

    Output
    ------
    float
        Interpolated intensity value.
    """
    height, width = image.shape

    if (
        x < 0
        or x > width - 1
        or y < 0
        or y > height - 1
    ):
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    alpha = x - x0
    beta = y - y0

    return (
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


def affine_matrix(
    angle,
    scale,
    tx,
    ty,
    center,
):
    """
    Construct an affine transformation matrix.

    Input
    -----
    angle : float
        Rotation angle in degrees.
    scale : float
        Uniform scaling factor.
    tx, ty : float
        Translation parameters.
    center : tuple
        Center of rotation.

    Output
    ------
    numpy.ndarray
        Homogeneous affine transformation matrix.
    """
    theta = np.deg2rad(angle)

    cosine = np.cos(theta)
    sine = np.sin(theta)

    cx, cy = center

    translate_to_origin = np.array(
        [
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1],
        ],
        dtype=float,
    )

    transform = np.array(
        [
            [
                scale * cosine,
                -scale * sine,
                0,
            ],
            [
                scale * sine,
                scale * cosine,
                0,
            ],
            [0, 0, 1],
        ],
        dtype=float,
    )

    translate_back = np.array(
        [
            [1, 0, cx + tx],
            [0, 1, cy + ty],
            [0, 0, 1],
        ],
        dtype=float,
    )

    return (
        translate_back
        @ transform
        @ translate_to_origin
    )


def inverse_warp(image, matrix):
    """
    Warp an image using inverse mapping.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    matrix : numpy.ndarray
        Forward transformation matrix.

    Output
    ------
    numpy.ndarray
        Warped image.
    """
    height, width = image.shape

    output = np.zeros_like(
        image,
        dtype=float,
    )

    inverse_matrix = np.linalg.inv(
        matrix
    )

    for y_destination in range(height):
        for x_destination in range(width):
            destination = np.array(
                [
                    x_destination,
                    y_destination,
                    1,
                ],
                dtype=float,
            )

            source = (
                inverse_matrix
                @ destination
            )

            x_source = source[0]
            y_source = source[1]

            output[
                y_destination,
                x_destination,
            ] = bilinear_interpolation(
                image,
                x_source,
                y_source,
            )

    return output


def create_image(size):
    """
    Create a synthetic test image.

    Input
    -----
    size : int
        Width and height of the image.

    Output
    ------
    numpy.ndarray
        Synthetic grayscale image.
    """
    image = np.zeros(
        (size, size),
        dtype=float,
    )

    image[
        size // 4:3 * size // 4,
        size // 4:3 * size // 4,
    ] = 0.35

    image[
        size // 3:2 * size // 3,
        size // 3:2 * size // 3,
    ] = 0.8

    y, x = np.indices(
        image.shape
    )

    radius = size // 10

    circle = (
        (x - size * 0.65) ** 2
        + (y - size * 0.35) ** 2
        <= radius**2
    )

    image[circle] = 1.0

    return image


def plot_results(
    original,
    warped,
    output_path,
):
    """
    Plot the image warping experiment.

    Input
    -----
    original : numpy.ndarray
        Original image.
    warped : numpy.ndarray
        Warped image.
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7, 3.5),
    )

    axes[0].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[1].imshow(
        warped,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Affine Warp"
    )

    for axis in axes:
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
    Run the image warping experiment.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figure.
    """
    size = 128

    image = create_image(
        size
    )

    center = (
        (size - 1) / 2,
        (size - 1) / 2,
    )

    matrix = affine_matrix(
        angle=25,
        scale=0.8,
        tx=10,
        ty=-5,
        center=center,
    )

    warped = inverse_warp(
        image,
        matrix,
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
        image,
        warped,
        output_directory
        / "image-warping.png",
    )


if __name__ == "__main__":
    main()