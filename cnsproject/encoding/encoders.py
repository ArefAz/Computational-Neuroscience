"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy import stats

import torch
from cnsproject.utils import make_gaussian


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    You will define the time duration into which you want to encode the data \
    as `time` and define the time resolution as `dt`. All computations will be \
    performed on the CPU by default. To handle computation on both GPU and CPU, \
    make sure to set the device as defined in `device` attribute to all your \
    tensors. You can add any other attributes to the child classes, if needed.

    The computation procedure should be implemented in the `__call__` method. \
    Data will be passed to this method as a tensor for further computations. You \
    might need to define more parameters for this method. The `__call__`  should return \
    the tensor of spikes with the shape (time_steps, \*population.shape).

    Arguments
    ---------
    sim_time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    """

    def __init__(
            self,
            sim_time: int,
            dt: Optional[float] = 1.0,
            device: Optional[str] = "cpu",
            data_max_val: Optional[int] = 255,
            **kwargs
    ) -> None:
        self.sim_time = sim_time
        self.dt = dt
        self.device = device
        self.data_max_val = data_max_val

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
    """

    def __init__(
            self,
            sim_time: int,
            dt: Optional[float] = 1.0,
            device: Optional[str] = "cpu",
            data_max_val: Optional[Union[float, int]] = 255,
            **kwargs
    ) -> None:
        super().__init__(
            sim_time=sim_time,
            dt=dt,
            device=device,
            data_max_val=data_max_val,
            **kwargs
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data: torch.Tensor = data.type(torch.FloatTensor)
        data_scaled = torch.round(
            data * (self.sim_time - 1) / self.data_max_val
        )
        encoded_data = torch.zeros(self.sim_time, *data_scaled.shape)
        for t in range(self.sim_time):
            encoded_data[t] = (data_scaled + t == self.sim_time - 1)

        return encoded_data


class PositionEncoder(AbstractEncoder):
    """
    Position coding.

    Implement Position coding.
    """

    def __init__(
            self,
            sim_time: int,
            n_neurons: int,
            gaussian_sigma: float,
            dt: Optional[float] = 1.0,
            device: Optional[str] = "cpu",
            data_max_val: Optional[int] = 255,
            **kwargs
    ) -> None:
        super().__init__(
            sim_time=sim_time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.data_max_val = data_max_val
        self.n_neurons = n_neurons
        self.g_sigma = gaussian_sigma
        self.time_thresh: float = kwargs.get('time_thresh', 0.95)
        self.g_offset: float = self.data_max_val / self.n_neurons / 2
        self.gaussians = self.create_gaussians()

    def create_gaussians(self):
        gaussians = []
        offset = self.g_offset
        means = np.linspace(offset, self.data_max_val + offset, self.n_neurons)
        for mean in means:
            gaussian = make_gaussian(self.data_max_val, mean, self.g_sigma,
                                     offset=0)
            gaussians.append(gaussian)

        return torch.tensor(gaussians)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        encoded_data = torch.zeros(len(self.gaussians), *data.ravel().shape)
        for i, gaussian in enumerate(self.gaussians):
            # Calculate the difference tensor (the intersection heights)
            s_t = 1 - gaussian[data.ravel().tolist()]
            s_t = torch.where(s_t < self.time_thresh, s_t, torch.ones(1) * -1)
            encoded_data[i] = s_t
        encoded_data = encoded_data.reshape(-1, *data.shape)
        encoded_data = torch.round(encoded_data * self.sim_time)
        encoded_data = torch.where(encoded_data > 0, encoded_data,
                                   torch.ones(1) * -1)

        time_encoded_data = torch.zeros(self.sim_time, len(self.gaussians),
                                        *data.shape)
        # Distribute the encoded data into zero-one spikes in time
        # encoded_data has the actual time of the spikes
        for t in range(self.sim_time):
            for i, gaussian in enumerate(self.gaussians):
                time_encoded_data[t][i] = (encoded_data[i] == t)

        return time_encoded_data


class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
            self,
            sim_time: int,
            rate_max: int,
            dt: Optional[float] = 1.0,
            device: Optional[str] = "cpu",
            data_max_val: Optional[Union[int, float]] = 255,
            **kwargs
    ) -> None:
        super().__init__(
            sim_time=sim_time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.rate_max = rate_max
        self.data_max_val = data_max_val

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data: torch.Tensor = data.type(torch.FloatTensor).to(self.device)
        r_x = data * self.rate_max / self.data_max_val
        r_x_dt = r_x * (self.dt / self.sim_time)
        encoded_data = torch.zeros(self.sim_time, *data.shape)
        p = torch.rand_like(encoded_data)
        for t in range(self.sim_time):
            encoded_data[t] = torch.less(p[t], r_x_dt)
        return encoded_data
