"""
Module for utility functions.

TODO.

Use this module to implement any required utility function.

Note: You are going to need to implement DoG and Gabor filters. A possible opt
ion would be to write them in this file but it is not a must and you can define\
a separate module/package for them.
"""

import torch
import numpy as np
import copy
import time
from cnsproject.network.neural_populations import NeuralPopulation, \
    LIFPopulation
from cnsproject.network.monitors import Monitor
from typing import List


def create_line(
        a: float,
        b: float,
        noise_std: float = 0.1,
        size: int = 1000
) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.standard_normal(size) * noise_std
    out = a * np.arange(size, dtype=np.float32) + b + noise
    return out


def apply_pulse_current(
        neuron: NeuralPopulation,
        monitor: Monitor,
        current: float,
        iters: int = 1000,
        zero_percent: int = 2
) -> None:
    zero_time = iters * zero_percent // 100
    for i in range(zero_time):
        neuron.forward(torch.tensor([0], dtype=torch.float32))
        monitor.record()
    for i in range(iters - zero_time):
        neuron.forward(torch.tensor([current], dtype=torch.float32))
        monitor.record()


def apply_random_current(
        neuron: NeuralPopulation,
        monitor: Monitor,
        line_params: dict,
        iters: int = 1000,
        zero_percent: int = 2,
) -> None:
    zero_time = iters * zero_percent // 100
    currents = create_line(line_params['a'], line_params['b'],
                           line_params['noise_std'], size=iters - zero_time)
    for i in range(zero_time):
        neuron.forward(torch.tensor([0], dtype=torch.float32))
        monitor.record()
    for current in currents:
        neuron.forward(torch.tensor([current], dtype=torch.float32))
        monitor.record()


def simulate_current(
        neuron: NeuralPopulation,
        monitor: Monitor,
        line_params: dict,
        iters: int = 1000,
        zero_percent: int = 2
) -> int:
    if line_params is None:
        line_params = {}
    with torch.no_grad():
        apply_random_current(neuron, monitor, line_params, iters,
                             zero_percent=zero_percent)
        spikes = monitor.get("s")
        # count the number of spikes (spike is True)
        spike_count = len(spikes[spikes])
        # to consider the zero time in calculating the frequency of spikes
        freq = spike_count / (1 - zero_percent / 100)

    return round(freq)


def run_simulation(
        neuron: NeuralPopulation,
        monitor: Monitor,
        input_line_params: List[dict],
        iters: int,
        save_monitor_states: bool = False,
        zero_percent: int = 2
) -> (np.ndarray, np.ndarray, list):
    frequencies = []
    currents = []
    monitor_states = []
    for line_params in input_line_params:
        neuron.reset_state_variables()
        monitor.reset_state_variables()
        f = simulate_current(neuron, monitor, line_params, iters,
                             zero_percent=zero_percent)
        frequencies.append(f)
        cur = line_params['b']
        currents.append(cur)
        if save_monitor_states:
            monitor_state = copy.deepcopy(monitor)
            monitor_states.append(monitor_state)

    frequencies = np.array(frequencies, dtype=np.uint16)
    currents = np.array(currents, dtype=np.float16)
    return frequencies, currents, monitor_states


def run_simulation_with_params(
        neuron_params: dict,
        monitor_vars: list,
        currents: np.ndarray,
        iters: int = 1000,
        save_monitor_states: bool = False,
        line_slop: float = 0,
        noise_std: float = 0,
        zero_percent: int = 2
) -> (np.ndarray, np.ndarray, Monitor, list, float):
    neuron = LIFPopulation((1,), **neuron_params)
    monitor = Monitor(neuron, state_variables=monitor_vars)

    input_line_params = [{'a': line_slop, 'b': b, 'noise_std': noise_std} for b
                         in currents]

    t1 = time.perf_counter()
    frequencies, currents, monitor_states = run_simulation(
        neuron,
        monitor,
        input_line_params,
        iters,
        save_monitor_states,
        zero_percent
    )
    t2 = time.perf_counter()
    simulation_time = t2 - t1
    return currents, frequencies, monitor, monitor_states, simulation_time


def round_of_rating(number: np.ndarray) -> np.ndarray:
    return (number * 2).round() / 2
