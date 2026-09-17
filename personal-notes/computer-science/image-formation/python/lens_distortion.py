"""
Radial lens distortion.

This script applies a radial distortion model to a regular grid and
illustrates the resulting geometric deformation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def distort_points(points, kappa_1, kappa_2):
    """
    Apply radial distortion to 2D points.

    Input
    -----
    points : array-like
        Undistorted two-dimensional coordinates.
    kappa_1 : float
        First radial distortion parameter.
    kappa_2 : float
        Second radial distortion parameter.

    Output
    ------
    numpy.ndarray
        Distorted two-dimensional coordinates.
    """
    x = points[:, 0]
    y = points[:, 1]

    r_squared = x**2 + y**2

    scale = (
        1
        + kappa_1 * r_squared
        + kappa_2 * r_squared**2
    )

    x_distorted = x * scale
    y_distorted = y * scale

    return np.column_stack(
        [x_distorted, y_distorted]
    )

def main():
    """
    Apply radial distortion to a regular grid.

    Input
    -----
    None

    Output
    ------
    None
        Saves the distorted grid as a figure.
    """
    coordinates = np.linspace(-1, 1, 100)

    kappa_1 = -0.2
    kappa_2 = 0.0

    for value in np.linspace(-1, 1, 9):
        horizontal = np.column_stack(
            [
                coordinates,
                np.full_like(coordinates, value),
            ]
        )

        vertical = np.column_stack(
            [
                np.full_like(coordinates, value),
                coordinates,
            ]
        )

        horizontal_distorted = distort_points(
            horizontal,
            kappa_1,
            kappa_2,
        )

        vertical_distorted = distort_points(
            vertical,
            kappa_1,
            kappa_2,
        )

        plt.plot(
            horizontal_distorted[:, 0],
            horizontal_distorted[:, 1],
        )

        plt.plot(
            vertical_distorted[:, 0],
            vertical_distorted[:, 1],
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "lens-distortion.png"
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