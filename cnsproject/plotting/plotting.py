"""
Module for visualization and plotting.

TODO.

Implement this module in any way you are comfortable with. You are free to use\
any visualization library you like. Providing live plotting and animations is\
also a bonus. The visualizations you will definitely need are as follows:

1. F-I curve.
2. Voltage/current dynamic through time.
3. Raster plot of spikes in a neural population.
4. Convolutional weight demonstration.
5. Weight change through time.
"""

import matplotlib.pyplot as plt
import numpy as np
from cnsproject.network.network import Monitor


def time_plot(monitor: Monitor) -> None:
    potential = monitor.get("potential")
    spikes = monitor.get("s")
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(potential.cpu())
    axs[1].plot(spikes.cpu(), c='r')
    f = spikes[spikes]
    print(len(f))


def fi_curve(f: np.ndarray, i: np.ndarray) -> None:
    if f.shape != i.shape:
        raise AssertionError('Input shapes have to match.')
    plt.plot(f, i)


if __name__ == '__main__':
    voltage = np.random.randn(100)
    frequency = np.random.randn(100)
    current = np.random.randn(100)
    time_plot(voltage)
    fi_curve(current, frequency)
