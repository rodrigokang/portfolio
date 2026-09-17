"""
Lambertian shading on a sphere.

This script computes the diffuse shading of a sphere under a
directional light source.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def lambertian_shading(normals, light_direction):
    """
    Compute Lambertian shading from surface normals.

    Input
    -----
    normals : array-like
        Unit surface normal vectors.
    light_direction : array-like
        Unit direction towards the light source.

    Output
    ------
    numpy.ndarray
        Lambertian intensity values.
    """
    light_direction = (
        light_direction
        / np.linalg.norm(light_direction)
    )

    intensity = normals @ light_direction

    return np.maximum(0, intensity)


def main():
    """
    Generate Lambertian shading on a sphere.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated shading image.
    """
    coordinates = np.linspace(-1, 1, 500)

    x, y = np.meshgrid(
        coordinates,
        coordinates,
    )

    r_squared = x**2 + y**2
    mask = r_squared <= 1

    z = np.zeros_like(x)
    z[mask] = np.sqrt(
        1 - r_squared[mask]
    )

    normals = np.stack(
        [x, y, z],
        axis=-1,
    )

    light_direction = np.array(
        [-1.0, 1.0, 1.0]
    )

    intensity = np.zeros_like(x)

    intensity[mask] = lambertian_shading(
        normals[mask],
        light_direction,
    )

    intensity[~mask] = np.nan

    plt.imshow(
        intensity,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="gray",
    )

    plt.axis("off")

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "lambertian-shading.png"
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