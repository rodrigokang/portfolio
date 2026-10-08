"""
Binary image processing.

This script applies morphological operations, connected-component
labeling, and a distance transform to a synthetic binary image.
"""

from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


def dilate(image, structuring_element):
    """
    Dilate a binary image.

    Input
    -----
    image : numpy.ndarray
        Input binary image.
    structuring_element : numpy.ndarray
        Binary structuring element.

    Output
    ------
    numpy.ndarray
        Dilated binary image.
    """
    height, width = image.shape

    element_height, element_width = (
        structuring_element.shape
    )

    pad_y = element_height // 2
    pad_x = element_width // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
    )

    output = np.zeros_like(
        image
    )

    for i in range(height):
        for j in range(width):
            neighborhood = padded[
                i:i + element_height,
                j:j + element_width,
            ]

            values = neighborhood[
                structuring_element == 1
            ]

            output[i, j] = np.max(
                values
            )

    return output


def erode(image, structuring_element):
    """
    Erode a binary image.

    Input
    -----
    image : numpy.ndarray
        Input binary image.
    structuring_element : numpy.ndarray
        Binary structuring element.

    Output
    ------
    numpy.ndarray
        Eroded binary image.
    """
    height, width = image.shape

    element_height, element_width = (
        structuring_element.shape
    )

    pad_y = element_height // 2
    pad_x = element_width // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
    )

    output = np.zeros_like(
        image
    )

    for i in range(height):
        for j in range(width):
            neighborhood = padded[
                i:i + element_height,
                j:j + element_width,
            ]

            values = neighborhood[
                structuring_element == 1
            ]

            output[i, j] = np.min(
                values
            )

    return output


def opening(image, structuring_element):
    """
    Apply morphological opening.

    Input
    -----
    image : numpy.ndarray
        Input binary image.
    structuring_element : numpy.ndarray
        Binary structuring element.

    Output
    ------
    numpy.ndarray
        Opened binary image.
    """
    return dilate(
        erode(
            image,
            structuring_element,
        ),
        structuring_element,
    )


def closing(image, structuring_element):
    """
    Apply morphological closing.

    Input
    -----
    image : numpy.ndarray
        Input binary image.
    structuring_element : numpy.ndarray
        Binary structuring element.

    Output
    ------
    numpy.ndarray
        Closed binary image.
    """
    return erode(
        dilate(
            image,
            structuring_element,
        ),
        structuring_element,
    )


def connected_components(image):
    """
    Label four-connected foreground components.

    Input
    -----
    image : numpy.ndarray
        Input binary image.

    Output
    ------
    numpy.ndarray
        Connected-component labels.
    """
    height, width = image.shape

    labels = np.zeros(
        image.shape,
        dtype=int,
    )

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    label = 0

    for i in range(height):
        for j in range(width):
            if (
                image[i, j] == 0
                or labels[i, j] != 0
            ):
                continue

            label += 1

            queue = deque([
                (i, j)
            ])

            labels[i, j] = label

            while queue:
                row, column = (
                    queue.popleft()
                )

                for di, dj in neighbors:
                    new_row = row + di
                    new_column = column + dj

                    if (
                        0 <= new_row < height
                        and 0 <= new_column < width
                        and image[
                            new_row,
                            new_column,
                        ] == 1
                        and labels[
                            new_row,
                            new_column,
                        ] == 0
                    ):
                        labels[
                            new_row,
                            new_column,
                        ] = label

                        queue.append(
                            (
                                new_row,
                                new_column,
                            )
                        )

    return labels


def distance_transform(image):
    """
    Compute the Euclidean distance transform.

    Input
    -----
    image : numpy.ndarray
        Input binary image.

    Output
    ------
    numpy.ndarray
        Distance to the nearest background pixel.
    """
    foreground = np.argwhere(
        image == 1
    )

    background = np.argwhere(
        image == 0
    )

    distances = np.zeros_like(
        image,
        dtype=float,
    )

    for row, column in foreground:
        differences = (
            background
            - np.array([
                row,
                column,
            ])
        )

        squared_distances = np.sum(
            differences**2,
            axis=1,
        )

        distances[row, column] = np.sqrt(
            np.min(
                squared_distances
            )
        )

    return distances


def create_image(size):
    """
    Create a synthetic binary image.

    Input
    -----
    size : int
        Width and height of the image.

    Output
    ------
    numpy.ndarray
        Synthetic binary image.
    """
    image = np.zeros(
        (size, size),
        dtype=int,
    )

    image[15:45, 15:45] = 1
    image[55:95, 65:105] = 1

    image[27:31, 27:31] = 0
    image[73:77, 83:87] = 0

    image[8:10, 75:77] = 1
    image[105:107, 25:27] = 1

    return image


def main():
    """
    Run the binary image processing experiment.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figure.
    """
    image = create_image(
        size=120
    )

    structuring_element = np.ones(
        (5, 5),
        dtype=int,
    )

    opened = opening(
        image,
        structuring_element,
    )

    cleaned = closing(
        opened,
        structuring_element,
    )

    labels = connected_components(
        cleaned
    )

    distances = distance_transform(
        cleaned
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Binary image"
    )

    axes[1].imshow(
        cleaned,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Morphological cleanup"
    )

    axes[2].imshow(
        labels,
        cmap="viridis",
    )

    axes[2].set_title(
        "Connected components"
    )

    axes[3].imshow(
        distances,
        cmap="viridis",
    )

    axes[3].set_title(
        "Distance transform"
    )

    for axis in axes:
        axis.axis("off")

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "binary-image-processing.png"
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