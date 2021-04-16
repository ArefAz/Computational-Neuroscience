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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cnsproject.network.network import Monitor
from cnsproject.network.neural_populations import ELIFPopulation, \
    AELIFPopulation, LIFPopulation


def time_plot(monitor: Monitor, figsize=(12.5, 15), plot_spikes=False,
              neuron_type=ELIFPopulation) -> None:
    if neuron_type == ELIFPopulation or neuron_type == AELIFPopulation:
        draw_theta_lines = True
    else:
        draw_theta_lines = False
    potential = monitor.get("potential").cpu()
    spikes = monitor.get("s").cpu()
    in_currents = monitor.get("in_current").cpu()
    u_rest = monitor.get("u_rest").cpu()[-1]
    threshold = monitor.get("threshold").cpu()[-1]
    tau = monitor.get("tau").cpu()[-1]

    if plot_spikes:
        fig, axs = plt.subplots(3, figsize=figsize)
    else:
        fig, axs = plt.subplots(2, figsize=figsize)

    axs[0].title.set_text("Neuron Dynamics with $\\tau={}$".format(tau))
    if draw_theta_lines:
        theta_reset = monitor.get("theta_reset").cpu()[-1]
        axs[0].axhline(y=theta_reset, color='y', linestyle='--',
                       label="$\\theta_{reset}$")
    axs[0].axhline(y=threshold, color='y', linestyle='--', label="$threshold$")
    if draw_theta_lines:
        theta_rh = monitor.get("theta_rh").cpu()[-1]
        axs[0].axhline(y=theta_rh, color='y', linestyle='--',
                       label="$\\theta_{rh}$")
    axs[0].axhline(y=u_rest, color='y', linestyle='--', label="$u_{rest}$")
    axs[0].set_ylabel('$u(t)$')
    spike_points = (
        np.where(spikes),
        np.ones_like(spikes[spikes]) * np.array(threshold)
    )
    axs[0].scatter(*spike_points, c='r', marker='*', s=20, label="$Spiked$")
    axs[0].plot(potential, linewidth=0.75)
    axs[0].legend()

    if plot_spikes:
        axs[1].title.set_text('Spikes')
        axs[1].set_ylabel('spike')
        axs[1].set_yticks([0, 1])
        axs[1].plot(spikes, c='r', linewidth=0.5)
        axs[2].title.set_text('Input')
        axs[2].set_xlabel('time')
        axs[2].set_ylabel('I(t)')
        axs[2].plot(in_currents, c='g', linewidth=0.5)
    else:
        axs[1].title.set_text('$Input$')
        axs[1].set_xlabel('$time$')
        axs[1].set_ylabel('$I(t)$')
        axs[1].plot(in_currents, c='g', linewidth=0.5)


def adaptation_plot(monitor: Monitor, figsize=(12.5, 5)) -> None:
    adaptation_factor = monitor.get('w')
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.title.set_text('Adaptation Curve')
    ax.plot(adaptation_factor)
    ax.set_xlabel("$time$")
    ax.set_ylabel("$w$")


def fi_curve(i: np.ndarray, f: np.ndarray, figsize=(12.5, 5)) -> None:
    if i.shape != f.shape:
        raise AssertionError('Input shapes have to match.')
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.title.set_text('$F-I$ Curve')
    ax.set_xlabel('$I(t)$')
    ax.set_ylabel('$f$ (1/s)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_yticks(f)
    ax.set_xticks(i.round(2))
    ax.grid()
    ax.plot(i, f, marker='o', mfc='r')
