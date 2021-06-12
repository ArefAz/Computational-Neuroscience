from abc import ABC, abstractmethod
from typing import Union, Sequence, Tuple, Optional
from torch.nn.modules.utils import _pair
from .neural_populations import NeuralPopulation, LIFPopulation
from ..filtering.filters import conv2d_tensor, max_pool2d

import torch
import numpy as np


class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    In case of learning, you will need to define the learning rate (`lr`) and the \
    learning rule to follow. Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `wmin` and `wmax`. \
    Synaptic strengths might decay in time and do not last forever. To define \
    the decay rate of the synaptic weights, use `weight_decay` attribute. Also, \
    if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__()

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        neuron_type = kwargs.get('neuron_type', LIFPopulation)
        self.pre: neuron_type = pre
        self.post: NeuralPopulation = post

        self.weight_decay = weight_decay

        from ..learning.learning_rules import NoOp

        learning_rule = kwargs.get('learning_rule', NoOp)

        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.wmin = kwargs.get('wmin', 0.)
        self.wmax = kwargs.get('wmax', 1.)
        self.norm = kwargs.get('norm', None)

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def compute(self) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", True)

        if learning:
            self.learning_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    Implement the dense connection pattern following the abstract connection\
    template.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.j0 = kwargs.get('j0', 0.5)
        self.sig0 = kwargs.get('sig0', 1e-10)
        self.wmin = torch.tensor(self.wmin)
        self.wmax = torch.tensor(self.wmax)
        mean = torch.tensor(self.j0 / self.pre.n, dtype=torch.float32)
        if mean < self.wmin or mean > self.wmax:
            raise ValueError('J0 is not compatible with wmax and wmin.')
        std = torch.tensor(self.sig0 / self.pre.n, dtype=torch.float32)
        # Initializing the weight_matrix with a normal distribution
        # with specified mean & std
        self.register_buffer(
            "w",
            torch.FloatTensor(*pre.shape, *post.shape).normal_(
                mean=mean,
                std=std
            )
        )

        # Make sure non of the weights are outside of [wmin, wmax]
        self.w = torch.where(self.w < self.wmin, mean,
                             self.w)
        self.w = torch.where(self.w > self.wmax, mean,
                             self.w)
        # Remove self connections
        if pre == post:
            self.w.fill_diagonal_(0)
        if self.w is None:
            # noinspection PyTypeChecker
            self.w: torch.Tensor = None
        # Negating all the weights if pre-synaptic population is inhibitory
        if self.pre.is_inhibitory:
            self.w *= -1

    def compute(self) -> None:
        # calculating the input of the post-synaptic neurons,
        # provided the weight_matrix and pre-synaptic neurons' spikes.
        # In the next line we multiply the spike states (of all pre-neurons)
        # in the weight_matrix (i.e., selecting the rows that the corresponding
        # neuron has spiked)
        additive_voltages: torch.Tensor = (self.w.T * self.pre.s).T
        # Then, here we add the effect of all spiked pre-synaptic neurons
        # for each post-synaptic neuron
        additive_voltages = additive_voltages.sum(dim=0)
        self.post.add_to_voltage(additive_voltages)

    def update(self, **kwargs) -> None:
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection\
    template.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            n_pre_connections: int = 10,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        if n_pre_connections > pre.n:
            raise ValueError(
                'number of pre-synapse connections cannot be greater than '
                'pre-pop #neurons'
            )
        weights_value = torch.tensor((self.wmax - self.wmin) / 2,
                                     dtype=torch.float32)
        # Initializing the weight_matrix with an arbitrary fixed value.
        # self.register_buffer(
        #     "weight_matrix",
        #     torch.ones(*pre.shape, *post.shape) * weights_value
        # )
        mean = kwargs.get("mean", 0.5)
        mean = torch.tensor(mean, dtype=torch.float32)
        self.register_buffer(
            "w",
            torch.FloatTensor(*pre.shape, *post.shape).normal_(
                mean=mean,
                std=0.15
            )
        )
        self.w = torch.where(self.w < self.wmin, mean,
                             self.w)
        self.w = torch.where(self.w > self.wmax, mean,
                             self.w)
        # First all of the possible connections' weights are set to
        # a fixed value, then the specified number them will be set back to
        # zero to have "num_pre_connections" non-zero weight in the
        # weight_matrix
        self.fill_weight_matrix(n_pre_connections)
        if self.w is None:
            # noinspection PyTypeChecker
            self.w: torch.Tensor = None
        if self.pre.is_inhibitory:
            self.w *= -1

    def fill_weight_matrix(self, num_pre_connections: int) -> None:
        for i in range(self.w.shape[1]):
            # randomly select non-connections
            zero_indexes = np.random.choice(
                self.pre.n,
                self.pre.n - num_pre_connections,
                replace=False
            )
            self.w[:, i][zero_indexes] = 0
        # Remove self connections
        if self.pre == self.post:
            self.w.fill_diagonal_(0)

    def compute(self) -> None:
        additive_voltages: torch.Tensor = (self.w.T * self.pre.s).T
        additive_voltages = additive_voltages.sum(dim=0)
        self.post.add_to_voltage(additive_voltages)

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class ConvolutionalConnection(AbstractConnection):
    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = 'valid',
            kernel_size: Union[int, Tuple[int, int]] = None,
            w: torch.Tensor = None,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.kernel_size = kernel_size
        if self.kernel_size is None:
            if w is None:
                raise ValueError(
                    "at least one of 'w' and 'kernel_size' should be "
                    "provided in ConvolutionalConnection arguments."
                )
            else:
                self.kernel_size = (w.shape[0], w.shape[1])
        self.kernel_size = _pair(self.kernel_size)
        self.strides = _pair(strides)
        self.padding = padding
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(
                    torch.rand(*self.kernel_size),
                    self.wmin,
                    self.wmax,
                )
            else:
                w = (self.wmax - self.wmin) * torch.rand(*self.kernel_size)
                w += self.wmin
                w = torch.clamp(
                    w,
                    self.wmin,
                    self.wmax,
                )
            w -= w.mean()
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.register_buffer("w", w)
        self.check_sizes()

    def check_sizes(self) -> None:
        test_in = torch.rand(*self.pre.shape)
        test_out = conv2d_tensor(test_in, self.w, self.padding, self.strides)
        for i, x in enumerate(self.post.shape):
            if test_out.shape[i] != x:
                raise ValueError(
                    "post-synaptic pop shape is not compatible with specified "
                    "convolution params\n expected {} but received {}.".format(
                        test_out.shape[i], x
                    )
                )

    def compute(self) -> None:
        conv_out = conv2d_tensor(
            self.pre.s.float(),
            self.w,
            padding=self.padding,
            strides=self.strides
        )
        self.post.add_to_voltage(conv_out)

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class PoolingConnection(AbstractConnection):
    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            kernel_size: Union[int, Tuple[int, int]] = 2,
            strides: Optional[Union[int, Tuple[int, int]]] = None,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            **kwargs
        )
        self.kernel_size = _pair(kernel_size)
        if strides is None:
            self.strides = self.kernel_size
        else:
            self.strides = _pair(strides)

        self.check_sizes()
        self.has_spiked = torch.zeros_like(self.post.s)

    def check_sizes(self) -> None:
        test_in = torch.rand(*self.pre.shape)
        test_out = max_pool2d(test_in, self.kernel_size, self.strides)
        for i, x in enumerate(self.post.shape):
            if test_out.shape[i] != x:
                raise ValueError(
                    "post-synaptic pop shape is not compatible with specified "
                    "convolution params\n expected {} but received {}.".format(
                        test_out.shape[i], x
                    )
                )

    def compute(self) -> None:
        max_out = max_pool2d(
            self.pre.s,
            kernel_size=self.kernel_size,
            strides=self.strides
        ).type(torch.BoolTensor)
        self.has_spiked = torch.logical_or(self.has_spiked, self.post.s)
        max_out = torch.where(self.has_spiked, False, max_out)
        self.post.add_to_voltage(max_out)

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.

        Note: You should be careful with this method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass
