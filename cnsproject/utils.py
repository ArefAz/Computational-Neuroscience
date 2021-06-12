import glob

import cv2
import torch
import numpy as np
import math
import copy
import time

from cnsproject.network.neural_populations import NeuralPopulation, \
    LIFPopulation, ELIFPopulation, AELIFPopulation
from cnsproject.network import Network
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection
from typing import List, Union, Iterable, Tuple


def make_gaussian(size, mu, sigma, normalize=False, offset=0):
    x = np.arange(-offset, size + offset, 1, np.float32)

    gaussian = np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))
    if normalize:
        norm = 1 / (sigma * math.sqrt(2 * math.pi))
        gaussian *= norm
    return gaussian


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


def run_connection_simulation(**kwargs):
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

    # inhibitory_currents = torch.zeros(sim_time, *inhibitory_pop.shape)
    # for i, currents in enumerate(all_currents):
    #     inhibitory_currents[i] = currents[inhibitory_input_samples]
    # net = Network(learning=False)
    # net.add_layer(excitatory_pop, "ep")
    # net.add_layer(inhibitory_pop, "ip")
    # net.add_connection(connection_1, "ep", "ep")
    # net.add_connection(connection_2, "ep", "ip")
    # net.add_connection(connection_3, "ip", "ep")
    # net.add_monitor(monitor_1, "ep")
    # net.add_monitor(monitor_2, "ip")
    # input_dict = dict(ep=all_currents, ip=inhibitory_currents)
    # net.reset_state_variables()
    # net.run(sim_time=sim_time, inputs=input_dict)

    return monitor_1, monitor_2


def run_decision_simulation(**kwargs):
    connection_types = kwargs.get('connection_types', [RandomConnection] * 3)
    n_connections = kwargs.get('num_connections', [20, 20, 20])
    w_maxes = kwargs.get('w_maxes', [1, 1, 1])
    n_neurons = kwargs.get('n_neurons', 1000)
    default_neuron_params = {'threshold': -50, 'u_rest': -65, 'tau': 50}
    neuron_params = kwargs.get('neuron_params', default_neuron_params)
    sim_time = kwargs.get('sim_time', 2000)
    zero_time = kwargs.get('zero_time', 200)
    dt = kwargs.get('dt', 1.0)
    neuron_noise_std = kwargs.get('neuron_noise_std', 4)
    time_noise_std = kwargs.get('time_noise_std', 5)
    input_base_value = kwargs.get('input_base_value', 20)
    sine_amplitude = kwargs.get('sine_amplitude', 0)
    sine_freq = kwargs.get('sine_freq', 1)
    step_amplitude = kwargs.get('step_amplitude', 0)
    ep1_add_linear = kwargs.get('ep1_add_linear', False)
    line_slop = kwargs.get('line_slop', 0)
    sine_abs = kwargs.get('sine_abs', True)
    j0 = kwargs.get('j0', [10, 10, 10])
    sig0 = kwargs.get('sig0', [0.2] * 3)

    ep1 = LIFPopulation(shape=(n_neurons * 4 // 10,),
                        is_inhibitory=False,
                        **neuron_params)
    ep2 = LIFPopulation(shape=(n_neurons * 4 // 10,),
                        is_inhibitory=False,
                        **neuron_params)
    ip = LIFPopulation(shape=(n_neurons * 2 // 10,),
                       is_inhibitory=True,
                       threshold=-60, u_rest=-65, tau=25)
    monitor_vars = ["s", "neuron_potentials", "in_current"]
    monitor_ep1 = Monitor(ep1, state_variables=monitor_vars)
    monitor_ep2 = Monitor(ep2, state_variables=monitor_vars)
    monitor_ip = Monitor(ip,
                         state_variables=[monitor_vars[0], monitor_vars[-1]])

    # EE connections
    connection_ep1_ep1 = connection_types[0](
        ep1, ep1, j0=j0[0], sig0=sig0[0], n_pre_connections=n_connections[0],
        wmax=w_maxes[0]
    )
    connection_ep2_ep2 = connection_types[0](
        ep2, ep2, j0=j0[0], sig0=sig0[0], n_pre_connections=n_connections[0],
        wmax=w_maxes[0]
    )

    # EI connections
    connection_ep1_ip = connection_types[1](
        ep1, ip, j0=j0[1], sig0=sig0[1], n_pre_connections=n_connections[1],
        wmax=w_maxes[1]
    )
    connection_ep2_ip = connection_types[1](
        ep2, ip, j0=j0[1], sig0=sig0[1], n_pre_connections=n_connections[1],
        wmax=w_maxes[1]
    )

    # IE connections
    connection_ip_ep1 = connection_types[2](
        ip, ep1, j0=j0[2], sig0=sig0[2], n_pre_connections=n_connections[2],
        wmax=w_maxes[2]
    )
    connection_ip_ep2 = connection_types[2](
        ip, ep2, j0=j0[2], sig0=sig0[2], n_pre_connections=n_connections[2],
        wmax=w_maxes[2]
    )

    monitor_ep1.set_time_steps(sim_time, dt)
    monitor_ep2.set_time_steps(sim_time, dt)
    monitor_ip.set_time_steps(sim_time, dt)
    monitor_ep1.reset_state_variables()
    monitor_ep2.reset_state_variables()
    monitor_ip.reset_state_variables()

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.manual_seed(0)
    time_noise = (torch.rand(sim_time, dtype=torch.float32) - 0.5
                  ) * time_noise_std
    neuron_noise = (torch.rand(
        sim_time,
        *ep1.shape
    ) - 0.5) * neuron_noise_std
    all_currents = torch.zeros(sim_time, *ep1.shape)
    for t in range(sim_time):
        if t < zero_time // 2 or t > sim_time - zero_time // 2:
            added_value = 0
        else:
            added_value = input_base_value
        currents = (
                torch.ones(*ep1.shape) * (added_value + time_noise[t])
        )
        currents += neuron_noise[t]
        all_currents[t] = currents

    ip_zero_input = torch.zeros(sim_time, *ip.shape)
    ep2_currents = torch.clone(all_currents)
    ep2_currents[zero_time * 2:-zero_time * 2] += step_amplitude
    sine = torch.sin(
        torch.arange(0, 10 * sine_freq * (sim_time // 1000), 0.01 * sine_freq)
    ) * sine_amplitude
    linear = torch.arange(len(ep2_currents),
                          dtype=torch.float32) - zero_time * 2
    linear *= float(line_slop)

    if sine_abs:
        sine = torch.abs(sine)
    sine[:zero_time * 2] = 0
    sine[-zero_time * 2:] = 0
    linear[:zero_time * 2] = 0
    linear[-zero_time * 2:] = 0
    for i in range(len(ep2_currents.T)):
        ep2_currents[..., i] += sine
        if not ep1_add_linear:
            ep2_currents[..., i] += linear
        else:
            all_currents[..., i] += linear

    net = Network(learning=False)
    net.add_layer(ep1, "ep1")
    net.add_layer(ep2, "ep2")
    net.add_layer(ip, "ip")
    net.add_connection(connection_ep1_ep1, "ep1", "ep1")
    net.add_connection(connection_ep2_ep2, "ep2", "ep2")
    net.add_connection(connection_ep1_ip, "ep1", "ip")
    net.add_connection(connection_ep2_ip, "ep2", "ip")
    net.add_connection(connection_ip_ep1, "ip", "ep1")
    net.add_connection(connection_ip_ep2, "ip", "ep2")
    net.add_monitor(monitor_ep1, "ep1")
    net.add_monitor(monitor_ep2, "ep2")
    net.add_monitor(monitor_ip, "ip")
    input_dict = dict(ep1=all_currents, ep2=ep2_currents, ip=ip_zero_input)
    net.reset_state_variables()
    net.run(sim_time=sim_time, current_inputs=input_dict)

    return monitor_ep1, monitor_ep2, monitor_ip


class SimpleRewardCalculator(object):
    def __init__(self, out_pop: NeuralPopulation, d_ratio: float,
                 d_values: list):
        self.out_pop = out_pop
        self.n = self.out_pop.n
        if self.n != 2:
            raise ValueError('Out-pop needs exactly 2 neurons.')
        self.d_ratio = d_ratio
        self.da_t = 0
        self.d_values = d_values
        self.sim_time: int = 0

    def set_sim_time(self, sim_time: int):
        self.sim_time = sim_time

    def calc_reward(self, t: int):
        spikes = self.out_pop.s
        pattern_id = self.find_pattern(t)
        if pattern_id == 0:
            if spikes[0] and not spikes[1]:
                self.da_t = self.d_values[0]
            elif spikes[0] and spikes[1]:
                self.da_t = self.d_values[1]
            elif not spikes[0] and spikes[1]:
                self.da_t = self.d_values[2]
            else:
                self.da_t = 0
        elif pattern_id == 1:
            if spikes[1] and not spikes[0]:
                self.da_t = self.d_values[0]
            elif spikes[1] and spikes[0]:
                self.da_t = self.d_values[1]
            elif not spikes[1] and spikes[0]:
                self.da_t = self.d_values[2]
            else:
                self.da_t = 0
        else:
            raise ValueError('Wrong in_pattern_id!')

        return self.da_t * self.d_ratio

    def get_da_t(self):
        return self.da_t * self.d_ratio

    def find_pattern(self, t):
        if self.sim_time == 0:
            raise RuntimeError('sim-time is not set!')
        if t % (self.sim_time // 10) < self.sim_time // 20:
            pattern_id = 0
        else:
            pattern_id = 1

        return pattern_id


def relu_normalize(in_tensor: torch.Tensor) -> torch.Tensor:
    out = torch.where(in_tensor > 0, in_tensor, torch.zeros(1))
    out /= out.max()
    return out


def read_images(path: str, size: Tuple[int, int] = (150, 150)) -> list:
    image_paths = sorted(glob.glob(f'{path}/*'))
    images = []
    for path in image_paths:
        image = cv2.imread(path)[..., ::-1]
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        image_gray = cv2.resize(image_gray, size)
        image_gray = image_gray.astype(float) / 255
        images.append(image_gray)

    return images
