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
import torch
from cnsproject.network.network import Monitor
from cnsproject.network.neural_populations import ELIFPopulation, \
    AELIFPopulation, LIFPopulation


def time_plot(monitor: Monitor, figsize=(12.5, 15), plot_spikes=False,
              neuron_type=LIFPopulation) -> None:
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


def raster_plot(monitor_excitatory: Monitor, monitor_inhibitory: Monitor,
                figsize=(15, 3), ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    spikes_excitatory = monitor_excitatory.get("s").cpu()
    spikes_inhibitory = monitor_inhibitory.get("s").cpu()
    x = np.arange(len(spikes_excitatory))
    n_neurons_ex = len(spikes_excitatory.T)
    n_neurons_in = len(spikes_inhibitory.T)
    for i in range(n_neurons_ex):
        y = np.where(spikes_excitatory[:, i], (i + 10), -10).astype(int)
        if i == 0:
            ax.scatter(x, y, color='b', s=0.75, label='$Excitatory$')
        else:
            ax.scatter(x, y, color='b', s=0.75)

    for i in range(n_neurons_in):
        y = np.where(spikes_inhibitory[:, i], (i + 15 + n_neurons_ex),
                     -10).astype(int)
        if i == 0:
            ax.scatter(x, y, color='r', s=0.75, label='$Inhibitory$')
        else:
            ax.scatter(x, y, color='r', s=0.75)
    ax.set_ylim(bottom=-1, top=n_neurons_ex + n_neurons_in + 20)
    ax.legend()
    ax.title.set_text('raster plot')
    ax.set_ylabel('neuron#')
    # ax.yaxis.set_visible(False)


def input_plot(monitor: Monitor, figsize=(15, 3), ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.suptitle('The input used for all the experiments')
    inputs = monitor.get("in_current").cpu()
    ax.plot(inputs[:, 0], linewidth=0.75, label="$neuron1_{{input}}$")
    ax.plot(inputs[:, 1], linewidth=0.75, label="$neuron2_{{input}}$")
    ax.plot(inputs[:, 3], linewidth=0.75, label="$neuron3_{{input}}$")
    ax.title.set_text('input plot')
    ax.set_xlabel('$time(ms)$')
    ax.set_ylabel('$I(t)$')
    ax.legend()


def activity_plot(monitor: Monitor, monitor_2: Monitor, figsize=(15, 3),
                  ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    spikes = monitor.get("s").cpu()
    x = np.arange(len(spikes))
    y = []
    n_neurons = len(spikes.T)
    for i in x:
        spikes_count = spikes[i].type(torch.IntTensor).sum() / n_neurons
        y.append(spikes_count)
    y = np.array(y)
    ax.plot(x, y, linewidth=0.5, c='b', label='$Excitatory$')
    area = np.trapz(y, dx=1)
    ax.plot([], [], c='b', linewidth=0.5, label="$AUC = {:.2f}$".format(area))

    spikes = monitor_2.get("s").cpu()
    x = np.arange(len(spikes))
    y = []
    n_neurons = len(spikes.T)
    for i in x:
        spikes_count = spikes[i].type(torch.IntTensor).sum() / n_neurons
        y.append(spikes_count)
    y = np.array(y)
    ax.plot(x, y, linewidth=0.3, c='r', label='$Inhibitory$')
    area = np.trapz(y, dx=1)
    ax.plot([], [], c='r', linewidth=0.3, label="$AUC = {:.2f}$".format(area))
    ax.legend()
    ax.title.set_text('activity plot')
    ax.set_ylabel('activation ratios')


def draw_connection_plots(m1: Monitor, m2: Monitor, figsize=(16, 8),
                          title: str = 'connection plots'):
    fig, ax = plt.subplots(2, figsize=figsize)
    fig.suptitle(title)
    raster_plot(m1, m2, ax=ax[0])
    activity_plot(m1, m2, ax=ax[1])
    ax[1].set_xlabel('$time(ms)$')