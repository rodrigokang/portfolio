"""
Aliasing produced by discrete sampling.

This script samples two sinusoidal signals with different frequencies
and illustrates how they can produce indistinguishable samples.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sinusoid(x, frequency):
    """
    Evaluate a sinusoidal signal.

    Input
    -----
    x : array-like
        Positions at which the signal is evaluated.
    frequency : float
        Frequency of the sinusoid.

    Output
    ------
    numpy.ndarray
        Signal values.
    """
    return np.sin(
        2 * np.pi * frequency * x
    )


def main():
    """
    Generate an example of aliasing.

    Input
    -----
    None

    Output
    ------
    None
        Saves the generated figure.
    """
    frequency_1 = 0.75
    frequency_2 = 1.25
    sampling_frequency = 2.0

    x = np.linspace(
        0,
        8,
        1000,
    )

    sample_x = np.arange(
        0,
        8,
        1 / sampling_frequency,
    )

    signal_1 = sinusoid(
        x,
        frequency_1,
    )

    signal_2 = sinusoid(
        x,
        frequency_2,
    )

    samples = sinusoid(
        sample_x,
        frequency_1,
    )

    plt.plot(
        x,
        signal_1,
        label="f = 0.75",
    )

    plt.plot(
        x,
        signal_2,
        label="f = 1.25",
    )

    plt.scatter(
        sample_x,
        samples,
        label="Samples",
    )

    plt.xlabel("x")
    plt.ylabel("s(x)")
    plt.legend()

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures"
        / "aliasing.png"
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