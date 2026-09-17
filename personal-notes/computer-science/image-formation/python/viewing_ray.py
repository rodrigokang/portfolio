"""
Viewing ray under perspective projection.

This script illustrates how different 3D points along the same viewing
ray project onto the same point on the image plane.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def project_point(point, focal_length):
    """
    Project a 3D point onto the 2D image plane.

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


def point_on_viewing_ray(image_point, depth, focal_length):
    """
    Compute a 3D point on the viewing ray of an image point.

    Input
    -----
    image_point : array-like
        Image coordinates (x, y).
    depth : float
        Depth Z of the desired 3D point.
    focal_length : float
        Focal length of the camera.

    Output
    ------
    numpy.ndarray
        Three-dimensional point (X, Y, Z) on the viewing ray.
    """
    x, y = image_point

    X = x * depth / focal_length
    Y = y * depth / focal_length

    return np.array([X, Y, depth])


def main():
    """
    Generate the viewing ray example and its figure.

    Input
    -----
    None

    Output
    ------
    None
        Prints the projected points and saves the generated figure.
    """
    focal_length = 5.0
    image_point = np.array([1.0, 0.5])
    depths = [10.0, 20.0, 30.0]

    points = np.array(
        [
            point_on_viewing_ray(
                image_point,
                depth,
                focal_length,
            )
            for depth in depths
        ]
    )

    # Verify that all 3D points have the same projection
    for point in points:
        projection = project_point(point, focal_length)
        print(f"{point} -> {projection}")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Camera centre
    camera = np.array([0.0, 0.0, 0.0])

    ax.scatter(
        camera[0],
        camera[1],
        camera[2],
        marker="o",
        label="Camera centre",
    )

    # Viewing ray
    ray_end = points[-1]

    ax.plot(
        [camera[0], ray_end[0]],
        [camera[1], ray_end[1]],
        [camera[2], ray_end[2]],
        label="Viewing ray",
    )

    # 3D points along the ray
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        marker="o",
        label="3D points",
    )

    # Virtual image plane at Z = f
    plane_size = 2.5

    X_plane, Y_plane = np.meshgrid(
        [-plane_size, plane_size],
        [-plane_size, plane_size],
    )

    Z_plane = np.full_like(X_plane, focal_length)

    ax.plot_surface(
        X_plane,
        Y_plane,
        Z_plane,
        alpha=0.2,
    )

    # Image point expressed as a 3D point on the virtual image plane
    image_point_3d = np.array(
        [
            image_point[0],
            image_point[1],
            focal_length,
        ]
    )

    ax.scatter(
        image_point_3d[0],
        image_point_3d[1],
        image_point_3d[2],
        marker="o",
        label="Image point",
    )

    # Labels for the 3D points
    for i, point in enumerate(points, start=1):
        ax.text(
            point[0],
            point[1],
            point[2],
            f"  P{i}",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "viewing-ray.png"
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