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
    LIFPopulation, ELIFPopulation, AELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection, \
    AbstractConnection
from typing import List, Union, Iterable


def create_line(
        a: float,
        b: float,
        noise_std: float = 0.1,
        size: int = 1000
) -> np.ndarray:
    noise = torch.randn(size, dtype=torch.float32) * noise_std
    out = a * torch.arange(size, dtype=torch.float32) + b + noise
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
        neuron.forward(torch.zeros(*neuron.shape, dtype=torch.float32))
        monitor.record()
    for i in range(iters - zero_time):
        neuron.forward(
            torch.full(*neuron.shape, fill_value=current, dtype=torch.float32))
        monitor.record()


def apply_random_current(
        neuron: NeuralPopulation,
        monitor: Monitor,
        line_params: dict,
        iters: int = 1000,
        zero_percent: int = 2,
) -> None:
    zero_time = iters * zero_percent // 2 // 100
    noise_std = line_params['noise_std']
    currents = create_line(line_params['a'], line_params['b'],
                           noise_std,
                           size=iters - 2 * zero_time)
    for i in range(zero_time):
        neuron.forward(
            torch.randn(*neuron.shape, dtype=torch.float32) * noise_std)
        monitor.record()
    for current in currents:
        neuron.forward(
            torch.ones(*neuron.shape, dtype=torch.float32) * current
        )
        monitor.record()
    for i in range(zero_time):
        neuron.forward(
            torch.randn(*neuron.shape, dtype=torch.float32) * noise_std)
        monitor.record()
    # for i in range(zero_time):
    #     neuron.forward(torch.tensor([10], dtype=torch.float32))
    #     monitor.record()


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
        freq = spike_count  # / (1 - zero_percent / 100)

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
        zero_percent: int = 2,
        neuron_type: Union[
            NeuralPopulation, LIFPopulation, ELIFPopulation, AELIFPopulation] =
        LIFPopulation,
        pop_shape: Iterable[int] = (1,),
) -> (np.ndarray, np.ndarray, Monitor, list, float):
    neuron = neuron_type(pop_shape, **neuron_params)
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


def run_connection_simulation(
        **kwargs,
):
    connection_type = kwargs.get('connection_type', RandomConnection)
    num_connections = kwargs.get('num_connections', [20, 30, 10])
    w_maxes = kwargs.get('w_maxes', [1, 1, 1])
    n_neurons = kwargs.get('n_neurons', 100)
    default_neuron_params = {'threshold': -50, 'u_rest': -65, 'tau': 50}
    neuron_params = kwargs.get('neuron_params', default_neuron_params)
    sim_time = kwargs.get('sim_time', 1000)
    zero_time = kwargs.get('zero_time', 100)
    dt = kwargs.get('dt', 1.0)
    neuron_noise_std = kwargs.get('neuron_noise_std', 0.1)
    time_noise_std = kwargs.get('time_noise_std', 0.5)
    input_base_value = kwargs.get('input_base_value', 20)
    j0 = kwargs.get('j0', [10, 10, 10])
    sig0 = kwargs.get('sig0', [1e-10, 1e-10, 1e-10])

    excitatory_pop = LIFPopulation(shape=(n_neurons * 8 // 10,),
                                   is_inhibitory=False,
                                   **neuron_params)
    inhibitory_pop = LIFPopulation(shape=(n_neurons * 2 // 10,),
                                   is_inhibitory=True,
                                   **neuron_params)
    monitor_vars = ["s", "neuron_potentials", "in_current"]
    monitor_1 = Monitor(excitatory_pop, state_variables=monitor_vars)
    monitor_2 = Monitor(inhibitory_pop, state_variables=["s"])
    connection_1 = connection_type(excitatory_pop, excitatory_pop,
                                   j0=j0[0], sig0=sig0[0],
                                   num_pre_connections=num_connections[0],
                                   wmax=w_maxes[0])
    connection_2 = connection_type(excitatory_pop, inhibitory_pop,
                                   j0=j0[1], sig0=sig0[1],
                                   num_pre_connections=num_connections[1],
                                   wmax=w_maxes[1])
    connection_3 = connection_type(inhibitory_pop, excitatory_pop,
                                   j0=j0[2], sig0=sig0[2],
                                   num_pre_connections=num_connections[2],
                                   wmax=w_maxes[2])
    monitor_1.set_time_steps(sim_time, dt)
    monitor_1.reset_state_variables()
    monitor_2.set_time_steps(sim_time, dt)
    monitor_2.reset_state_variables()

    np.random.seed(0)
    torch.manual_seed(0)
    time_noise = torch.randn(sim_time, dtype=torch.float32) * time_noise_std
    neuron_noise = torch.randn(
        sim_time,
        *excitatory_pop.shape
    ) * neuron_noise_std
    all_currents = torch.zeros(sim_time, *excitatory_pop.shape)
    for t in range(sim_time):
        if t < zero_time // 2 or t > sim_time - zero_time // 2:
            added_value = 0
        else:
            added_value = input_base_value
        currents = (torch.ones(*excitatory_pop.shape) * (
                added_value + time_noise[t]))
        currents += neuron_noise[t]
        all_currents[t] = currents

    inhibitory_input_samples = np.random.choice(
        len(all_currents[0]),
        inhibitory_pop.n,
    )
    for currents in all_currents:
        connection_1.compute()
        excitatory_pop.forward(currents)
        connection_2.compute()
        inhibitory_pop.forward(currents[inhibitory_input_samples])
        connection_3.compute()
        monitor_1.record()
        monitor_2.record()

    return monitor_1, monitor_2
