"""
Module for neuronal dynamics and populations.
"""

from functools import reduce
from abc import abstractmethod
from operator import mul
from typing import Union, Iterable, Sequence
from cnsproject.filtering.filters import make_gaussian

import torch
import torch.nn.functional as F


class NeuralPopulation(torch.nn.Module):
    """
    Base class for implementing neural populations.

    Make sure to implement the abstract methods in your child class. Note that this template\
    will give you homogeneous neural populations in terms of excitations and inhibitions. You\
    can modify this by removing `is_inhibitory` and adding another attribute which defines the\
    percentage of inhibitory/excitatory neurons or use a boolean tensor with the same shape as\
    the population, defining which neurons are inhibitory.

    The most important attribute of each neural population is its `shape` which indicates the\
    number and/or architecture of the neurons in it. When there are connected populations, each\
    pre-synaptic population will have an impact on the post-synaptic one in case of spike. This\
    spike might be persistent for some duration of time and with some decaying magnitude. To\
    handle this coincidence, four attributes are defined:
    - `spike_trace` is a boolean indicating whether to record the spike trace in each time step.
    - `additive_spike_trace` would indicate whether to save the accumulated traces up to the\
        current time step.
    - `tau_s` will show the duration by which the spike trace persists by a decaying manner.
    - `trace_scale` is responsible for the scale of each spike at the following time steps.\
        Its value is only considered if `additive_spike_trace` is set to `True`.

    Make sure to call `reset_state_variables` before starting the simulation to allocate\
    and/or reset the state variables such as `s` (spikes tensor) and `traces` (trace of spikes).\
    Also do not forget to set the time resolution (dt) for the simulation.

    Each simulation step is defined in `forward` method. You can use the utility methods (i.e.\
    `compute_potential`, `compute_spike`, `refractory_and_reset`, and `compute_decay`) to break\
    the differential equations into smaller code blocks and call them within `forward`. Make\
    sure to call methods `forward` and `compute_decay` of `NeuralPopulation` in child class\
    methods; As it provides the computation of spike traces (not necessary if you are not\
    considering the traces). The `forward` method can either work with current or spike trace.\
    You can easily work with any of them you wish. When there are connected populations, you\
    might need to consider how to convert the pre-synaptic spikes into current or how to\
    change the `forward` block to support spike traces as input.

    There are some more points to be considered further:
    - Note that parameters of the neuron are not specified in child classes. You have to\
        define them as attributes of the corresponding class (i.e. in __init__) with suitable\
        naming.
    - In case you want to make simulations on `cuda`, make sure to transfer the tensors\
        to the desired device by defining a `device` attribute or handling the issue from\
        upstream code.
    - Almost all variables, parameters, and arguments in this file are tensors with a\
        single value or tensors of the shape equal to population`s shape. No extra\
        dimension for time is needed. The time dimension should be handled in upstream\
        code and/or monitor objects.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    is_inhibitory : False, Optional
        Whether the neurons are inhibitory or excitatory. The default is False.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    # noinspection PyTypeChecker
    def __init__(
            self,
            shape: Sequence[int],
            spike_trace: bool = True,
            additive_spike_trace: bool = True,
            tau_s: Union[float, torch.Tensor] = 15.,
            trace_scale: Union[float, torch.Tensor] = 1.,
            is_inhibitory: bool = False,
            learning: bool = True,
            **kwargs
    ) -> None:
        super().__init__()

        if len(shape) == 2:
            self.shape = (1, *shape)
        else:
            self.shape = shape
        self.n = reduce(mul, self.shape)
        self.spike_trace = spike_trace
        self.additive_spike_trace = additive_spike_trace

        if self.spike_trace:
            # You can use `torch.Tensor()` instead of `torch.zeros(*shape)`
            # if `reset_state_variables`
            # is intended to be called before every simulation.
            self.register_buffer("traces", torch.zeros(*self.shape))
            self.register_buffer("tau_s", torch.tensor(tau_s))

            if self.additive_spike_trace:
                self.register_buffer("trace_scale", torch.tensor(trace_scale))

            self.register_buffer("trace_decay", torch.ones_like(self.tau_s))

        self.is_inhibitory = is_inhibitory
        self.learning = learning
        self.do_exp_decay: bool = kwargs.get("do_exp_decay", False)

        # You can use `torch.Tensor()` instead of
        # `torch.zeros(*shape, dtype=torch.bool)` if
        # `reset_state_variables` is intended to be
        # called before every simulation.
        self.register_buffer("s", torch.zeros(*self.shape, dtype=torch.bool))

        if self.s is None:
            self.s: torch.Tensor = None
            self.traces: torch.Tensor = None
            self.tau_s: torch.Tensor = None
            self.trace_scale: torch.Tensor = None
            self.trace_decay: torch.Tensor = None
        self.neuron_potentials: torch.Tensor

    @abstractmethod
    def forward(self, traces: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        traces : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        if self.spike_trace:
            if self.do_exp_decay:
                self.traces += -(self.dt / self.tau_s) * self.traces
            else:
                self.traces *= self.trace_decay

            if self.additive_spike_trace:
                self.traces += self.trace_scale * self.s
            else:
                self.traces.masked_fill_(self.s, 1)

    @abstractmethod
    def compute_potential(self, in_current: torch.Tensor) -> None:
        """
        Compute the potential of neurons in the population.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_spike(self) -> None:
        """
        Compute the spike tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Refractor and reset the neurons.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Set the decays.

        Returns
        -------
        None

        """
        # self.dt = self.dt.clone().detach()

        if self.spike_trace:
            self.trace_decay = torch.exp(-self.dt / self.tau_s)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        self.s.zero_()

        if self.spike_trace:
            self.traces.zero_()

    def train(self, mode: bool = True) -> "NeuralPopulation":
        """
        Set the population's training mode.

        Parameters
        ----------
        mode : bool, optional
            Mode of training. `True` turns on the training while `False` turns\
            it off. The default is True.

        Returns
        -------
        NeuralPopulation

        """
        self.learning = mode
        return super().train(mode)


class LIFPopulation(NeuralPopulation):
    """
    Layer of Leaky Integrate and Fire neurons.

    Implement LIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    # noinspection PyTypeChecker
    def __init__(
            self,
            shape: Sequence[int],
            spike_trace: bool = True,
            additive_spike_trace: bool = True,
            tau_s: Union[float, torch.Tensor] = 15.,
            trace_scale: Union[float, torch.Tensor] = 1.,
            is_inhibitory: bool = False,
            learning: bool = True,
            kwta: int = None,
            **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
            **kwargs
        )
        self._init_kwargs(kwargs)
        self.kwta = kwta
        self.register_buffer(
            "neuron_potentials",
            torch.ones(*self.shape, dtype=torch.float32) * self.u_rest
        )
        self.register_buffer(
            "in_current",
            torch.zeros(*self.shape, dtype=torch.float32)
        )

        self.initialize_lateral(**kwargs)
        self.selected_maps = []
        # just to silence PyCharm's warning
        if self.neuron_potentials is None:
            self.neuron_potentials: torch.Tensor = None
        if self.s is None:
            self.s: torch.Tensor = None
        if self.tau is None:
            self.tau: torch.Tensor = None
        if self.r is None:
            self.r: torch.Tensor = None
        if self.in_current is None:
            self.in_current: torch.Tensor = None

    def _init_kwargs(self, kwargs: dict) -> None:
        if (k := 'threshold') in kwargs:
            tensor = torch.tensor(kwargs[k], dtype=torch.float32)
        else:
            tensor = torch.tensor(-55, dtype=torch.float32)
        self.register_buffer('threshold', tensor)
        if (k := 'u_rest') in kwargs:
            tensor = torch.tensor(kwargs[k], dtype=torch.float32)
        else:
            tensor = torch.tensor(-60, dtype=torch.float32)
        self.register_buffer('u_rest', tensor)
        if self.u_rest >= self.threshold:
            raise ValueError('u_rest should be lower than threshold')
        if (k := 'dt') in kwargs:
            tensor = torch.tensor(kwargs[k], dtype=torch.float32)
        else:
            tensor = torch.tensor(1, dtype=torch.float32)
        self.register_buffer('dt', tensor)
        if (k := 'r') in kwargs:
            tensor = torch.tensor(float(kwargs[k]), dtype=torch.float32)
        else:
            tensor = torch.tensor(1, dtype=torch.float32)
        self.register_buffer('r', tensor)
        if (k := 'tau') in kwargs:
            tensor = torch.tensor(float(kwargs[k]), dtype=torch.float32)
        else:
            tensor = torch.tensor(15, dtype=torch.float32)
        self.register_buffer('tau', tensor)

    def initialize_lateral(self, **kwargs):
        inh_int: float = kwargs.get("lat_inh_int", 0.5)
        k: int = kwargs.get("lat_k_size", 3)
        if k % 2 == 0:
            raise ValueError("lat_k_size should be an odd integer.")
        sigma: float = kwargs.get("lat_sigma", 3)
        g_inv = -make_gaussian(k, sigma)
        inhibit_mask = torch.tensor(g_inv, dtype=torch.float32) * inh_int
        inhibit_maps = torch.zeros(*self.shape)
        self.register_buffer("inhibit_mask", inhibit_mask)
        self.register_buffer("inhibit_maps", inhibit_maps)
        inh_kernel = torch.stack((self.inhibit_mask,) * len(inhibit_maps))
        self.register_buffer("inh_kernel", inh_kernel)

    def forward(self, traces: torch.Tensor) -> None:
        self.in_current = traces
        self.compute_potential(traces)
        self.compute_spike()
        self.refractory_and_reset()
        self.compute_decay()
        super().forward(traces)

    def compute_potential(self, in_current: torch.Tensor) -> None:
        u_update = -(self.dt / self.tau) * (
                (self.neuron_potentials - self.u_rest) - (self.r * in_current)
        )
        self.neuron_potentials += u_update

    def compute_spike(self) -> None:

        if self.kwta is not None:
            volts = self.neuron_potentials

            k = self.inh_kernel.shape[-1]
            p = k // 2

            winners_idx = [
                (i, *divmod(f.argmax().item(), f.shape[0]))
                for i, f in enumerate(volts)
                if i not in self.selected_maps and f.max() >= self.threshold
            ]

            if len(winners_idx) + len(self.selected_maps) >= self.kwta:
                winners_idx.sort(key=lambda x: volts[x], reverse=True)
                winners_idx = winners_idx[:self.kwta - len(self.selected_maps)]

            if len(winners_idx) and len(self.selected_maps) <= self.kwta:
                winners = torch.zeros_like(self.s)
                for idx in winners_idx:
                    winners[idx] = True
                    self.selected_maps.append(idx[0])

                self.s = winners

                inh_map = F.conv2d(
                    winners.unsqueeze(0).float(),
                    self.inh_kernel.unsqueeze(0),
                    padding=p
                ).squeeze()
                self.add_to_voltage(inh_map)
            else:
                self.s.zero_()

        else:
            self.s = torch.where(
                self.neuron_potentials < self.threshold,
                False,
                True
            )

    def refractory_and_reset(self, index: int = 0) -> None:
        # noinspection PyTypeChecker
        self.neuron_potentials = torch.where(
            self.neuron_potentials < self.threshold,
            self.neuron_potentials,
            self.u_rest
        )

    def compute_decay(self) -> None:
        super().compute_decay()

    def reset_state_variables(self):
        super().reset_state_variables()
        self.neuron_potentials.fill_(self.u_rest)
        self.selected_maps.clear()

    def add_to_voltage(self, v: torch.Tensor) -> None:
        if self.neuron_potentials.shape[-2:] != v.shape[-2:]:
            raise ValueError(
                "input tensor shape should be compatible the "
                "neuron_potentials' shape, but got {} and {}".format(
                    self.neuron_potentials.shape, v.shape)
            )

        self.neuron_potentials += v


class InputPopulation(NeuralPopulation):
    """
    Neural population for user-defined spike pattern.

    This class is implemented for future usage. Extend it if needed.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
            self,
            shape: Sequence[int],
            spike_trace: bool = True,
            additive_spike_trace: bool = True,
            tau_s: Union[float, torch.Tensor] = 15.,
            trace_scale: Union[float, torch.Tensor] = 1.,
            learning: bool = True,
            **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            learning=learning,
            **kwargs
        )

    def forward(self, traces: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        traces : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        self.s = traces
        super().compute_decay()
        super().forward(traces)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()


class ELIFPopulation(NeuralPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.

    Implement ELIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.

    Note: You can use LIFPopulation as parent class as well.
    """

    def __init__(
            self,
            shape: Sequence[int],
            spike_trace: bool = True,
            additive_spike_trace: bool = True,
            tau_s: Union[float, torch.Tensor] = 10.,
            trace_scale: Union[float, torch.Tensor] = 1.,
            is_inhibitory: bool = False,
            learning: bool = True,
            **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        self._init_kwargs(kwargs)

        self.register_parameter(
            "potential",
            torch.nn.parameter.Parameter(self.u_rest, requires_grad=False)
        )
        self.register_buffer(
            "in_current",
            torch.zeros((1,), dtype=torch.float32)
        )
        self.spiked = False

        # just to silence PyCharm's warning
        if self.potential is None:
            self.potential = None
        if self.s is None:
            self.s = None
        if self.tau is None:
            self.tau = None
        if self.r is None:
            self.r = None
        if self.in_current is None:
            self.in_current = None

    def _init_kwargs(self, kwargs: dict) -> None:
        dtype = torch.float32
        tensor = torch.tensor(kwargs.get('threshold', -50), dtype=dtype)
        self.register_buffer('threshold', tensor)
        tensor = torch.tensor(kwargs.get('u_rest', -65), dtype=dtype)
        self.register_buffer('u_rest', tensor)
        self.dt = torch.tensor(kwargs.get('dt', 1), dtype=dtype)
        tensor = torch.tensor(kwargs.get('r', 1), dtype=dtype)
        self.register_buffer('r', tensor)
        tensor = torch.tensor(kwargs.get('tau', 50), dtype=dtype)
        self.register_buffer('tau', tensor)
        tensor = torch.tensor(kwargs.get('theta_rh', -55), dtype=dtype)
        self.register_buffer('theta_rh', tensor)
        tensor = torch.tensor(kwargs.get('theta_reset', -20), dtype=dtype)
        self.register_buffer('theta_reset', tensor)
        if not (self.u_rest < self.theta_rh < self.threshold <
                self.theta_reset):
            raise ValueError(
                'Incorrect parameters encountered. Check threshold '
                'u_rest, theta_rh and theta_reset'
            )
        tensor = torch.tensor(kwargs.get('delta_T', 1), dtype=dtype)
        self.register_buffer('delta_T', tensor)

    def forward(self, traces: torch.Tensor) -> None:
        self.in_current = traces
        self.compute_potential(traces)
        self.compute_spike()
        super().forward(traces)
        self.compute_decay()

    def compute_potential(self, in_current: torch.Tensor) -> None:

        exp = torch.exp((self.potential.data - self.theta_rh) / self.delta_T)
        if self.delta_T == 0:
            exp = torch.zeros_like(exp)
        u_update = (self.dt / self.tau) * (
                -(self.potential.data - self.u_rest) + self.delta_T * exp +
                (self.r * in_current)
        )
        self.potential.data = self.potential.data + u_update

    def compute_spike(self) -> None:
        if self.potential.data[0] >= self.threshold and not self.spiked:
            self.s = torch.tensor(True, dtype=torch.bool)
            self.spiked = True
        else:
            self.s = torch.tensor(False, dtype=torch.bool)

        if self.potential.data[0] >= self.theta_reset:
            self.refractory_and_reset()

    def refractory_and_reset(self) -> None:
        self.spiked = False
        self.potential = torch.nn.parameter.Parameter(
            torch.unsqueeze(self.u_rest, 0)
        )

    def compute_decay(self) -> None:
        super().compute_decay()

    def reset_state_variables(self):
        super().reset_state_variables()
        self.potential.data = self.u_rest
        self.spiked = False
        self.s = torch.tensor(False, dtype=torch.bool)


class AELIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Exponential Leaky Integrate and Fire neurons.

    Implement adaptive ELIF neural dynamics(Parameters of the model must be\
    modifiable). Follow the template structure of NeuralPopulation class for\
    consistency.

    Note: You can use ELIFPopulation as parent class as well.
    """

    def __init__(
            self,
            shape: Sequence[int],
            spike_trace: bool = True,
            additive_spike_trace: bool = True,
            tau_s: Union[float, torch.Tensor] = 10.,
            trace_scale: Union[float, torch.Tensor] = 1.,
            is_inhibitory: bool = False,
            learning: bool = True,
            **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        self._init_kwargs(kwargs)

        self.register_parameter(
            "potential",
            torch.nn.parameter.Parameter(self.u_rest, requires_grad=False)
        )
        self.register_parameter(
            'w',
            torch.nn.parameter.Parameter(
                torch.zeros((1,), dtype=torch.float32),
                requires_grad=False
            )
        )
        self.register_buffer(
            "in_current",
            torch.zeros((1,), dtype=torch.float32)
        )
        self.spiked = False
        # just to silence PyCharm's warning
        if self.potential is None:
            self.potential = None
        if self.s is None:
            self.s = None
        if self.tau is None:
            self.tau = None
        if self.r is None:
            self.r = None
        if self.in_current is None:
            self.in_current = None

    def _init_kwargs(self, kwargs: dict) -> None:
        dtype = torch.float32
        tensor = torch.tensor(kwargs.get('a', 1e-1), dtype=dtype)
        self.register_buffer('a', tensor)
        tensor = torch.tensor(kwargs.get('b', 1e-1), dtype=dtype)
        self.register_buffer('b', tensor)
        tensor = torch.tensor(kwargs.get('tau_w', 1), dtype=dtype)
        self.register_buffer('tau_w', tensor)
        tensor = torch.tensor(kwargs.get('threshold', -50), dtype=dtype)
        self.register_buffer('threshold', tensor)
        tensor = torch.tensor(kwargs.get('u_rest', -65), dtype=dtype)
        self.register_buffer('u_rest', tensor)
        self.dt = torch.tensor(kwargs.get('dt', 1), dtype=dtype)
        tensor = torch.tensor(kwargs.get('r', 1), dtype=dtype)
        self.register_buffer('r', tensor)
        tensor = torch.tensor(kwargs.get('tau', 50), dtype=dtype)
        self.register_buffer('tau', tensor)
        tensor = torch.tensor(kwargs.get('theta_rh', -55), dtype=dtype)
        self.register_buffer('theta_rh', tensor)
        tensor = torch.tensor(kwargs.get('theta_reset', -20), dtype=dtype)
        self.register_buffer('theta_reset', tensor)
        if not (self.u_rest < self.theta_rh < self.threshold <
                self.theta_reset):
            raise ValueError(
                'Incorrect parameters encountered. Check threshold '
                'u_rest, theta_rh and theta_reset'
            )
        tensor = torch.tensor(kwargs.get('delta_T', 1), dtype=dtype)
        self.register_buffer('delta_T', tensor)

    def forward(self, traces: torch.Tensor) -> None:
        self.in_current = traces
        self.compute_potential(traces)
        self.compute_spike()
        super().forward(traces)
        self.compute_decay()

    def compute_potential(self, in_current: torch.Tensor) -> None:

        self.compute_adaptation()
        exp = torch.exp((self.potential.data - self.theta_rh) / self.delta_T)
        if self.delta_T == 0:
            exp = torch.zeros_like(exp)
        u_update = (self.dt / self.tau) * (
                -(self.potential.data - self.u_rest) + (self.delta_T * exp) -
                (self.r * self.w.data) + (self.r * in_current)
        )
        self.potential.data = self.potential.data + u_update

    def compute_adaptation(self) -> None:
        w_update = (self.dt / self.tau_w) * (
                self.a * (self.potential.data - self.u_rest) - self.w.data +
                self.b * self.tau_w * int(self.s)
        )
        self.w.data = self.w.data + w_update

    def compute_spike(self) -> None:
        if self.potential.data[0] >= self.threshold and not self.spiked:
            self.s = torch.tensor(True, dtype=torch.bool)
            self.spiked = True
        else:
            self.s = torch.tensor(False, dtype=torch.bool)

        if self.potential.data[0] >= self.theta_reset and self.spiked:
            self.spiked = False
            self.refractory_and_reset()

    def refractory_and_reset(self) -> None:
        self.potential = torch.nn.parameter.Parameter(
            torch.unsqueeze(self.u_rest, 0)
        )

    def compute_decay(self) -> None:
        super().compute_decay()

    def reset_state_variables(self):
        super().reset_state_variables()
        self.potential.data = self.u_rest
        self.spiked = False
        self.w.data = torch.zeros((1,), dtype=torch.float32)
        self.s = torch.tensor(False, dtype=torch.bool)
