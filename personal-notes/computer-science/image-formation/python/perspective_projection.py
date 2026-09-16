"""
Perspective projection using the pinhole camera model.

This script implements perspective projection and illustrates how the
apparent size of an object changes with its depth relative to the camera.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def project_point(point, focal_length):
    """
    Project a 3D point onto a 2D image plane using the pinhole camera model.

    Input
    -----
    point : array-like
        Three-dimensional point (X, Y, Z).
    focal_length : float
        Focal length of the camera.

    Output
    ------
    numpy.ndarray
        Projected image coordinates (x, y).
    """
    X, Y, Z = point

    x = focal_length * X / Z
    y = focal_length * Y / Z

    return np.array([x, y])


def project_square(square, depth, focal_length):
    """
    Project a square at a given depth onto the 2D image plane.

    Input
    -----
    square : array-like
        Coordinates (X, Y) of the square vertices.
    depth : float
        Depth Z of the square.
    focal_length : float
        Focal length of the camera.

    Output
    ------
    numpy.ndarray
        Projected coordinates of the square vertices.
    """
    return np.array(
        [
            project_point((X, Y, depth), focal_length)
            for X, Y in square
        ]
    )


def main():
    """
    Run the perspective projection example and generate the figure.

    Input
    -----
    None

    Output
    ------
    None
        Prints projected coordinates and saves the generated figure.
    """
    focal_length = 5.0

    # Numerical example
    P = np.array([2.0, 1.0, 10.0])
    P_prime = np.array([2.0, 1.0, 20.0])

    p = project_point(P, focal_length)
    p_prime = project_point(P_prime, focal_length)

    print(f"P  = {P} -> p  = {p}")
    print(f"P' = {P_prime} -> p' = {p_prime}")

    # Square in the XY plane
    square = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    depths = [5.0, 10.0, 20.0]

    for depth in depths:
        projected_square = project_square(
            square,
            depth,
            focal_length,
        )

        plt.plot(
            projected_square[:, 0],
            projected_square[:, 1],
            marker="o",
            label=f"Z = {depth:g}",
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()

    # Save figure
    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "perspective-depth.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()