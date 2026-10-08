"""
Two-dimensional Fourier analysis.

This script visualizes the Fourier spectra of sinusoidal images
and demonstrates low-pass filtering in the frequency domain.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_sinusoid(size, frequency_x, frequency_y):
    """
    Create a two-dimensional sinusoidal image.

    Input
    -----
    size : int
        Width and height of the image.
    frequency_x : float
        Horizontal spatial frequency.
    frequency_y : float
        Vertical spatial frequency.

    Output
    ------
    numpy.ndarray
        Sinusoidal grayscale image.
    """
    coordinates = np.arange(size)

    x, y = np.meshgrid(
        coordinates,
        coordinates,
    )

    image = np.cos(
        2 * np.pi
        * (
            frequency_x * x
            + frequency_y * y
        )
    )

    return image


def magnitude_spectrum(image):
    """
    Compute the centered Fourier magnitude spectrum.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.

    Output
    ------
    numpy.ndarray
        Logarithmic Fourier magnitude spectrum.
    """
    transform = np.fft.fft2(
        image
    )

    shifted = np.fft.fftshift(
        transform
    )

    spectrum = np.log(
        1 + np.abs(shifted)
    )

    return spectrum


def low_pass_filter(image, radius):
    """
    Apply an ideal circular low-pass filter.

    Input
    -----
    image : numpy.ndarray
        Input grayscale image.
    radius : float
        Radius of the frequency-domain filter.

    Output
    ------
    numpy.ndarray
        Low-pass filtered image.
    """
    transform = np.fft.fft2(
        image
    )

    shifted = np.fft.fftshift(
        transform
    )

    height, width = image.shape

    center_y = height // 2
    center_x = width // 2

    y, x = np.ogrid[
        :height,
        :width,
    ]

    distance = np.sqrt(
        (x - center_x) ** 2
        + (y - center_y) ** 2
    )

    mask = (
        distance <= radius
    )

    filtered_transform = (
        shifted * mask
    )

    unshifted = np.fft.ifftshift(
        filtered_transform
    )

    filtered = np.fft.ifft2(
        unshifted
    )

    return np.real(
        filtered
    )


def plot_spectra(output_path):
    """
    Plot sinusoidal images and their Fourier spectra.

    Input
    -----
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    configurations = [
        (0.04, 0.00),
        (0.10, 0.00),
        (0.07, 0.07),
    ]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9, 6),
    )

    for column, (
        frequency_x,
        frequency_y,
    ) in enumerate(configurations):
        image = create_sinusoid(
            size=128,
            frequency_x=frequency_x,
            frequency_y=frequency_y,
        )

        spectrum = magnitude_spectrum(
            image
        )

        axes[0, column].imshow(
            image,
            cmap="gray",
        )

        axes[0, column].set_title(
            rf"$({frequency_x}, {frequency_y})$"
        )

        axes[1, column].imshow(
            spectrum,
            cmap="gray",
        )

        axes[0, column].axis("off")
        axes[1, column].axis("off")

    axes[0, 0].set_ylabel(
        "Image"
    )

    axes[1, 0].set_ylabel(
        "Spectrum"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_low_pass_filter(output_path):
    """
    Demonstrate low-pass frequency filtering.

    Input
    -----
    output_path : pathlib.Path
        Output figure path.

    Output
    ------
    None
        Saves the generated figure.
    """
    low_frequency = create_sinusoid(
        size=128,
        frequency_x=0.03,
        frequency_y=0.00,
    )

    high_frequency = create_sinusoid(
        size=128,
        frequency_x=0.18,
        frequency_y=0.00,
    )

    image = (
        low_frequency
        + 0.5 * high_frequency
    )

    filtered = low_pass_filter(
        image,
        radius=10,
    )

    original_spectrum = magnitude_spectrum(
        image
    )

    filtered_spectrum = magnitude_spectrum(
        filtered
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7, 7),
    )

    axes[0, 0].imshow(
        image,
        cmap="gray",
    )

    axes[0, 0].set_title(
        "Original"
    )

    axes[0, 1].imshow(
        original_spectrum,
        cmap="gray",
    )

    axes[0, 1].set_title(
        "Original spectrum"
    )

    axes[1, 0].imshow(
        filtered,
        cmap="gray",
    )

    axes[1, 0].set_title(
        "Low-pass filtered"
    )

    axes[1, 1].imshow(
        filtered_spectrum,
        cmap="gray",
    )

    axes[1, 1].set_title(
        "Filtered spectrum"
    )

    for axis in axes.flat:
        axis.axis("off")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    """
    Run the Fourier experiments.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figures.
    """
    output_directory = (
        Path(__file__).resolve().parent.parent
        / "figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_spectra(
        output_directory
        / "fourier-spectra.png"
    )

    plot_low_pass_filter(
        output_directory
        / "fourier-low-pass.png"
    )


if __name__ == "__main__":
    main()