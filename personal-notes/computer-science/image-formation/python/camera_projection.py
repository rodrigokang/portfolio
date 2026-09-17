"""
Projection of 3D points onto a 2D image plane.

This script transforms a 3D cube from world coordinates to camera
coordinates and projects it using the pinhole camera model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def project_points(points, R, t, focal_length):
    """
    Project 3D world points onto the 2D image plane.

    Input
    -----
    points : array-like
        Three-dimensional points in world coordinates.
    R : array-like
        Rotation matrix from world to camera coordinates.
    t : array-like
        Translation vector from world to camera coordinates.
    focal_length : float
        Focal length of the camera.

    Output
    ------
    numpy.ndarray
        Projected two-dimensional coordinates.
    """
    camera_points = (R @ points.T).T + t

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    x = focal_length * X / Z
    y = focal_length * Y / Z

    return np.column_stack([x, y])


def main():
    """
    Project a 3D cube and generate its 2D representation.

    Input
    -----
    None

    Output
    ------
    None
        Saves the projected cube as a figure.
    """
    cube = np.array(
        [
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
        ],
        dtype=float,
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    theta = np.radians(25)

    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t = np.array([0.0, 0.0, 6.0])

    focal_length = 5.0

    projected = project_points(
        cube,
        R,
        t,
        focal_length,
    )

    for i, j in edges:
        plt.plot(
            [projected[i, 0], projected[j, 0]],
            [projected[i, 1], projected[j, 1]],
        )

    plt.scatter(
        projected[:, 0],
        projected[:, 1],
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "camera-projection.png"
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