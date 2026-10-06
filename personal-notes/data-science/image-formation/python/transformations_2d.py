"""
Basic 2D geometric transformations.

This script applies translation, rigid, similarity, affine, and
projective transformations to a square.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def transform_points(points, H):
    """
    Transform 2D points using a homogeneous transformation matrix.

    Input
    -----
    points : array-like
        Two-dimensional coordinates of the points.
    H : array-like
        Homogeneous transformation matrix.

    Output
    ------
    numpy.ndarray
        Transformed two-dimensional coordinates.
    """
    points_h = np.column_stack(
        [points, np.ones(len(points))]
    )

    transformed_h = (H @ points_h.T).T

    transformed = (
        transformed_h[:, :2]
        / transformed_h[:, 2, np.newaxis]
    )

    return transformed


def main():
    """
    Apply several 2D transformations to a square.

    Input
    -----
    None

    Output
    ------
    None
        Saves a figure comparing the transformed squares.
    """
    square = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    theta = np.radians(30)

    translation = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    rigid = np.array(
        [
            [np.cos(theta), -np.sin(theta), 2.0],
            [np.sin(theta), np.cos(theta), 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    similarity = np.array(
        [
            [1.5 * np.cos(theta), -1.5 * np.sin(theta), 2.0],
            [1.5 * np.sin(theta), 1.5 * np.cos(theta), 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    affine = np.array(
        [
            [1.0, 0.5, 2.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    projective = np.array(
        [
            [1.0, 0.2, 2.0],
            [0.1, 1.0, 1.0],
            [0.15, 0.1, 1.0],
        ]
    )

    transformations = {
        "Translation": translation,
        "Rigid": rigid,
        "Similarity": similarity,
        "Affine": affine,
        "Projective": projective,
    }

    plt.plot(
        square[:, 0],
        square[:, 1],
        marker="o",
        label="Original",
    )

    for name, H in transformations.items():
        transformed = transform_points(square, H)

        plt.plot(
            transformed[:, 0],
            transformed[:, 1],
            marker="o",
            label=name,
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "transformations-2d.png"
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