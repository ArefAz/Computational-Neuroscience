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

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import List
from scipy.signal import savgol_filter
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


def raster_plot(monitor_excitatory: Monitor,
                monitor_inhibitory: Monitor = None,
                title: str = None,
                is_inhibitory: bool = False,
                event_plot: bool = False,
                figsize=(15, 3), ax=None, y_label: bool = True,
                legend: bool = True, set_xlim: bool = True,
                y_offset: int = 15, size: float = 0.75):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    spikes_excitatory = monitor_excitatory.get("s").cpu()
    x = np.arange(len(spikes_excitatory))
    n_neurons_ex = len(spikes_excitatory.T)
    s = np.array(spikes_excitatory).T
    s = np.argwhere(s)

    if event_plot:
        positions = [s[s[:, 0] == i][:, 1] for i in range(n_neurons_ex)]
        ax.eventplot(positions, colors='C0',
                     linestyles="-."
                     )
        label = "$Inhibitory$" if is_inhibitory else '$Excitatory$'
        ax.scatter([], [], color='C0', label=label, s=size)
        if monitor_inhibitory is not None:
            spikes_inhibitory = monitor_inhibitory.get("s").cpu()
            n_neurons_in = len(spikes_inhibitory.T)
            s = np.array(spikes_inhibitory).T
            s = np.argwhere(s)
            offset = n_neurons_ex + 10
            positions = [s[s[:, 0] == i - offset][:, 1] for i in
                         range(n_neurons_in + offset)]
            ax.eventplot(positions, colors='C1', lineoffsets=1,
                         linelengths=0.75)
            label = "$Inhibitory$"
            ax.scatter([], [], color='C1', label=label, s=size)
    else:
        for i in range(n_neurons_ex):
            y = np.where(spikes_excitatory[:, i], (i + y_offset), -10).astype(
                int)
            label = '$Excitatory$' if legend else None
            if i == 0:
                ax.scatter(x, y, color='C0', s=size, label=label)
            else:
                ax.scatter(x, y, color='C0', s=size)
        n_neurons_in = 0
        if monitor_inhibitory is not None:
            spikes_inhibitory = monitor_inhibitory.get("s").cpu()
            n_neurons_in = len(spikes_inhibitory.T)
            for i in range(n_neurons_in):
                y = np.where(spikes_inhibitory[:, i], (i + 15 + n_neurons_ex),
                             -10).astype(int)
                label = '$Inhibitory$' if legend else None
                if i == 0:
                    ax.scatter(x, y, color='C1', s=size, label=label)
                else:
                    ax.scatter(x, y, color='C1', s=size)
        ax.set_ylim(bottom=-1, )

    if set_xlim:
        ax.set_xlim([0, len(spikes_excitatory)])
    if legend:
        ax.legend()
    if title is None:
        title = 'raster plot'
    ax.title.set_text(title)
    if y_label:
        ax.set_ylabel('neuron#')
    # ax.yaxis.set_visible(False)


def input_plot(monitor: Monitor, figsize=(15, 3), draw_additional=True,
               ax=None, y_label: bool = True):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.suptitle('The input used for all the experiments')
    inputs = monitor.get("in_current").cpu()
    ax.plot(inputs[:, 0], linewidth=0.75, label="$neuron1_{{input}}$")
    if draw_additional:
        ax.plot(inputs[:, 1], linewidth=0.75, label="$neuron2_{{input}}$")
        ax.plot(inputs[:, 3], linewidth=0.75, label="$neuron3_{{input}}$")
    ax.title.set_text('input plot')
    ax.set_xlabel('$time(ms)$')
    if y_label:
        ax.set_ylabel('$I(t)$')
    ax.legend()


def activity_plot(monitor_excitatory: Monitor,
                  monitor_inhibitory: Monitor = None,
                  is_inhibitory: bool = False,
                  figsize=(15, 3),
                  smooth_size: int = 23, y_label: bool = True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    spikes = monitor_excitatory.get("s").cpu()
    x = np.arange(len(spikes))
    y = []
    n_neurons = len(spikes.T)
    for i in x:
        spikes_count = spikes[i].type(torch.IntTensor).sum() / n_neurons
        y.append(spikes_count)
    y = np.array(y)
    y = savgol_filter(y, smooth_size, 3)
    label = "$Inhibitory$" if is_inhibitory else '$Excitatory$'
    ax.plot(x, y, linewidth=0.5, c='C0', label=label)
    area = np.trapz(y, dx=1)
    ax.plot([], [], c='C0', linewidth=0.5, label="$AUC = {:.2f}$".format(area))

    if monitor_inhibitory is not None:
        spikes = monitor_inhibitory.get("s").cpu()
        x = np.arange(len(spikes))
        y = []
        n_neurons = len(spikes.T)
        for i in x:
            spikes_count = spikes[i].type(torch.IntTensor).sum() / n_neurons
            y.append(spikes_count)
        y = np.array(y)
        ax.plot(x, y, linewidth=0.3, c='C1', label='$Inhibitory$')
        area = np.trapz(y, dx=1)
        ax.plot([], [], c='C1', linewidth=0.3,
                label="$AUC = {:.2f}$".format(area))
    ax.set_yticks(np.linspace(0, 1.5, 7) / 10)
    ax.set_ylim([0, 0.15])
    ax.legend()
    ax.title.set_text('activity plot')
    if y_label:
        ax.set_ylabel('activation ratios')


def raster_plot_encoded_data(data: torch.Tensor, figsize=None, title=None,
                             ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if title is None:
        title = 'Raster Plot of Encoded Data'
    ax.title.set_text(title)
    if len(data.shape) == 3:
        n_neurons = data.shape[1] * data.shape[2]
    elif len(data.shape) == 4:
        n_neurons = data.shape[1] * data.shape[2] * data.shape[3]
    else:
        n_neurons = data.shape[1]
    data = data.reshape(-1, n_neurons)
    s = np.array(data).T
    s = np.argwhere(s)

    positions = [s[s[:, 0] == i][:, 1] for i in range(n_neurons)]
    ax.eventplot(positions, colors='C0', linestyles="dotted")
    # ax.set_ylim(bottom=-5)
    x_offset = int(len(data) * 0.05) // 2
    ax.set_xlim(left=-x_offset, right=len(data) + x_offset)


def activity_plot_encoded_data(data: torch.Tensor, figsize=None,
                               smooth_size=23, title=None, ax=None):
    if title is None:
        title = 'Encoded Data Activity Plot'
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.title.set_text(title)
    x = np.arange(len(data))
    y = []
    if len(data.shape) == 3:
        n_neurons = data.shape[1] * data.shape[2]
    elif len(data.shape) == 4:
        n_neurons = data.shape[1] * data.shape[2] * data.shape[3]
    else:
        n_neurons = data.shape[1]
    for i in x:
        spikes_count = data[i].type(torch.IntTensor).sum() / n_neurons
        y.append(spikes_count)
    y = np.array(y)
    if smooth_size != 0:
        y = savgol_filter(y, smooth_size, 3)
    ax.plot(x, y, c='C0', linewidth=0.75)
    ax.set_yticks([])


def draw_connection_plots(m1: Monitor, m2: Monitor, figsize=(16, 8),
                          title: str = 'connection plots'):
    fig, ax = plt.subplots(2, figsize=figsize)
    fig.suptitle(title)
    raster_plot(m1, m2, ax=ax[0])
    activity_plot(m1, m2, ax=ax[1])
    ax[1].set_xlabel('$time(ms)$')


def draw_decision_plots(monitors: List[Monitor], monitor_names: List[str],
                        figsize=(15, 15),
                        title: str = 'decision plots',
                        event_plot: bool = True,
                        smooth_size: int = 23):
    fig, ax = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(title)
    for i, monitor in enumerate(monitors):
        is_inhibitory = "IP" in monitor_names[i]
        if i == 0:
            y_label = True
        else:
            y_label = False
        raster_plot(monitor, ax=ax[0, i], title=monitor_names[i],
                    event_plot=event_plot, is_inhibitory=is_inhibitory,
                    y_label=y_label)
        activity_plot(monitor, ax=ax[1, i], is_inhibitory=is_inhibitory,
                      smooth_size=smooth_size, y_label=y_label)
        input_plot(monitor, ax=ax[2, i], draw_additional=True, y_label=y_label)


def draw_encoded_data_plots(encoded_data: torch.Tensor, smooth_size=13,
                            figsize=None, title=None, histogram=None):
    if title is None:
        title = ''
    if histogram is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(title)
    fig.tight_layout()
    if histogram is None:
        raster_plot_encoded_data(encoded_data, ax=ax)
        # ax.set_ylim(bottom=-0.25)
    else:
        raster_plot_encoded_data(encoded_data, ax=ax[0])
    if histogram is not None:
        activity_plot_encoded_data(encoded_data, smooth_size=smooth_size,
                                   ax=ax[1])
        flipped = torch.flip(encoded_data, dims=(0,))
        activity_plot_encoded_data(flipped, smooth_size=smooth_size, ax=ax[2],
                                   title='Mirrored Activity Plot')
        ax[3].plot(histogram, linewidth=0.75)
        ax[3].title.set_text('Original Image Histogram')
        ax[3].set_yticks([])


def draw_encoded_images():
    pass
