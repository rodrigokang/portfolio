"""
Point operations on a grayscale image.

This script applies bias, gain, gamma transformation, and histogram
equalization to a synthetic grayscale image.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def apply_gain_bias(image, gain, bias):
    """
    Apply gain and bias to an image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    gain : float
        Multiplicative gain.
    bias : float
        Additive bias.

    Output
    ------
    numpy.ndarray
        Transformed image.
    """
    result = gain * image + bias

    return np.clip(result, 0, 1)


def apply_gamma(image, gamma):
    """
    Apply a gamma transformation to an image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    gamma : float
        Gamma exponent.

    Output
    ------
    numpy.ndarray
        Gamma-transformed image.
    """
    return image**gamma


def histogram_equalization(image, levels=256):
    """
    Equalize the histogram of a grayscale image.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    levels : int
        Number of discrete intensity levels.

    Output
    ------
    numpy.ndarray
        Histogram-equalized image.
    """
    discrete = np.floor(
        image * (levels - 1)
    ).astype(int)

    histogram = np.bincount(
        discrete.ravel(),
        minlength=levels,
    )

    cdf = np.cumsum(histogram) / image.size

    return cdf[discrete]


def main():
    """
    Compare several point operations.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated comparison figure.
    """
    coordinates = np.linspace(0, 1, 500)

    x, y = np.meshgrid(
        coordinates,
        coordinates,
    )

    image = (
        0.15
        + 0.45 * x
        + 0.15 * np.sin(6 * np.pi * x)
        * np.sin(6 * np.pi * y)
    )

    image = np.clip(image, 0, 1)

    brighter = apply_gain_bias(
        image,
        gain=1.0,
        bias=0.2,
    )

    higher_contrast = apply_gain_bias(
        image,
        gain=1.4,
        bias=0.0,
    )

    gamma_corrected = apply_gamma(
        image,
        gamma=0.5,
    )

    equalized = histogram_equalization(
        image,
    )

    images = [
        image,
        brighter,
        higher_contrast,
        gamma_corrected,
        equalized,
    ]

    titles = [
        "Original",
        "Bias",
        "Gain",
        "Gamma",
        "Histogram equalization",
    ]

    figure, axes = plt.subplots(
        1,
        len(images),
        figsize=(15, 3),
    )

    for axis, result, title in zip(
        axes,
        images,
        titles,
    ):
        axis.imshow(
            result,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axis.set_title(title)
        axis.axis("off")

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "point-operations.png"
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