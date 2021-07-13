"""
Module for learning rules.
"""

from typing import Union, Optional, Sequence
from abc import ABC
from .rewards import AbstractReward
from ..network.connections import AbstractConnection, ConvolutionalConnection

import numpy as np
import torch.nn.functional as F
import torch


class LearningRule(ABC):
    """
    Abstract class for defining learning rules.

    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.

    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.connection = connection
        self.lr = torch.tensor(lr, dtype=torch.float)
        self.weight_decay = 1 - weight_decay if weight_decay else 1.

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        if (
                self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.w.clamp_(self.connection.wmin,
                                     self.connection.wmax)


class NoOp(LearningRule):
    """
    Learning rule with no effect.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.

        Returns
        -------
        None

        """
        super().update()


class STDP(LearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.is_flat: bool = kwargs.get("is_flat", False)
        self.weight_dependant: bool = kwargs.get("weight_dependant", False)
        # noinspection PyTypeChecker
        self.update_mask: torch.Tensor = None
        self.pre_traces = torch.zeros_like(self.connection.pre.traces)
        self.post_traces = torch.zeros_like(self.connection.post.traces)
        if isinstance(self.connection, ConvolutionalConnection):
            self.is_conv = True
        else:
            self.is_conv = False
        self.pre = self.connection.pre
        self.post = self.connection.post

    def update(self, **kwargs) -> None:
        if self.is_conv:
            self._conv_connection_update(**kwargs)
        else:
            self._dense_connection_update(**kwargs)
        super().update()

    def _dense_connection_update(self, **kwargs):
        if self.update_mask is None:
            self.update_mask = torch.ones(*self.connection.w.T.shape).bool()
        self.pre_traces = self.pre.traces
        self.post_traces = self.post.traces
        if self.is_flat:
            self.pre_traces = torch.where(self.pre_traces > 0.05, 1, 0)
            self.post_traces = torch.where(self.post_traces > 0.05, 1, 0)
        pre_s = self.pre.s
        post_s = self.post.s

        pre_spiked = (self.update_mask * pre_s).T
        if self.lr[0]:
            update_pre_s = -self.lr[0] * self.post_traces * pre_spiked
        else:
            update_pre_s = 0.

        post_spiked = (self.update_mask.T * post_s).T
        if self.lr[1]:
            update_post_s = self.lr[1] * self.pre_traces * post_spiked
            update_post_s = update_post_s.T
        else:
            update_post_s = 0.

        update_matrix = update_pre_s + update_post_s
        self.connection.w += update_matrix

    def _conv_connection_update(self, **kwargs):
        num_filters, _, kernel_height, kernel_width = self.connection.w.shape
        stride = self.connection.stride
        padding = self.connection.padding

        pre_s = F.unfold(
            self.pre.s.unsqueeze(0).float(),
            (kernel_height, kernel_width),
            padding=padding,
            stride=stride,
        ).swapdims(2, 1).squeeze()
        pre_traces = F.unfold(
            self.pre.traces.unsqueeze(0),
            kernel_size=(kernel_height, kernel_width),
            padding=padding,
            stride=stride
        ).swapdims(2, 1).squeeze()

        post_s = self.post.s.view(num_filters, -1).float()
        post_traces = self.post.traces.view(num_filters, -1)

        if self.is_flat:
            pre_traces = torch.where(pre_traces > 0.05, 1., 0.)
            post_traces = torch.where(post_traces > 0.05, 1., 0.)

        if self.lr[0]:
            pre = torch.mm(post_traces, pre_s)
            update_pre_s = -self.lr[0] * pre.view(self.connection.w.shape)
        else:
            update_pre_s = 0.

        if self.lr[1]:
            post = torch.mm(post_s, pre_traces)
            update_post_s = self.lr[1] * post.view(self.connection.w.shape)
        else:
            update_post_s = 0.

        self.connection.w += update_pre_s + update_post_s


class RSTDP(LearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.is_flat: bool = kwargs.get("is_flat", False)
        # noinspection PyTypeChecker
        self.update_mask: torch.Tensor = None
        self.pre_traces = torch.zeros_like(self.connection.pre.traces)
        self.post_traces = torch.zeros_like(self.connection.post.traces)
        self.tau_c = kwargs.get("tau_c", 500.)
        self.tau_c = torch.tensor(self.tau_c, dtype=torch.float32)
        # noinspection PyTypeChecker
        self.c: torch.Tensor = None

    def update(self, **kwargs) -> None:
        if self.c is None:
            self.c = torch.zeros_like(self.connection.w)
        reward: AbstractReward = kwargs.get("reward")
        if reward is None:
            raise RuntimeError("Reward should be passed to RSTDP's update!")
        d = reward.get_d()
        if self.update_mask is None:
            self.update_mask = torch.ones(*self.connection.w.T.shape).bool()
        self.pre_traces = self.connection.pre.traces
        self.post_traces = self.connection.post.traces

        pre_s = self.connection.pre.s
        post_s = self.connection.post.s

        pre_spiked = (self.update_mask * pre_s).T
        update_pre_s = -self.lr[0] * self.post_traces * pre_spiked
        post_spiked = (self.update_mask.T * post_s).T
        update_post_s = self.lr[1] * self.pre_traces * post_spiked
        update_post_s = update_post_s.T

        stdp_update_matrix = update_pre_s + update_post_s
        self.c += -(self.c / self.tau_c) + stdp_update_matrix
        self.connection.w += self.c * d
        super().update()


class FlatSTDP(LearningRule):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        pass


class FlatRSTDP(LearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    # noinspection PyTypeChecker
    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.update_mask: torch.Tensor = None
        self.pre_traces = torch.zeros_like(self.connection.pre.traces)
        self.post_traces = torch.zeros_like(self.connection.post.traces)
        self.c: torch.Tensor = None

    def update(self, **kwargs) -> None:
        if self.c is None:
            self.c = torch.zeros_like(self.connection.w)

        reward: AbstractReward = kwargs.get("reward")
        if reward is None:
            raise RuntimeError("Reward should be passed to RSTDP's update!")
        d = reward.get_d()
        if self.update_mask is None:
            self.update_mask = torch.ones(*self.connection.w.T.shape).bool()
        self.pre_traces = self.connection.pre.traces
        self.post_traces = self.connection.post.traces
        self.pre_traces = torch.where(self.pre_traces > 0.05, 1, 0)
        self.post_traces = torch.where(self.post_traces > 0.05, 1, 0)

        pre_s = self.connection.pre.s
        post_s = self.connection.post.s

        pre_spiked = (self.update_mask * pre_s).T
        update_pre_s = -self.lr[0] * self.post_traces * pre_spiked
        post_spiked = (self.update_mask.T * post_s).T
        update_post_s = self.lr[1] * self.pre_traces * post_spiked
        update_post_s = update_post_s.T

        flat_stdp_update_matrix = update_pre_s + update_post_s
        self.c = flat_stdp_update_matrix
        self.connection.w += self.c * d
        super().update()
