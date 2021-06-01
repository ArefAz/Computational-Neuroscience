from abc import ABC, abstractmethod

import torch


class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    Make sure to implement the abstract methods in your child class.

    To implement your dopamine functionality, You will write a class \
    inheriting this abstract class. You can add attributes to your \
    child class. The dynamics of dopamine function (DA) will be \
    implemented in `compute` method. So you will call `compute` in \
    your reward-modulated learning rules to retrieve the dopamine \
    value in the desired time step. To reset or update the defined \
    attributes in your reward function, use `update` method and \
    remember to call it your learning rule computations in the \
    right place.
    """

    def __init__(self, **kwargs):
        tau_d = kwargs.get("tau_d")
        if tau_d is None:
            raise RuntimeError('tau_d is a mandatory argument for reward!')
        self.d = torch.zeros(1, dtype=torch.float32)
        self.tau_d = torch.tensor(tau_d, dtype=torch.float32)

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass

    def get_d(self):
        return self.d


class SimpleReward(AbstractReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.da_t = None

    def compute(self, **kwargs) -> None:
        self.da_t = torch.tensor(kwargs.get("da_t", 0.))
        self.d += -(self.d / self.tau_d) + self.da_t

    def update(self, **kwargs) -> None:
        pass


class FlatReward(AbstractReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.da_t = None

    def compute(self, **kwargs) -> None:
        self.da_t = torch.tensor(kwargs.get("da_t", 0.))
        self.d = self.da_t

    def update(self, **kwargs) -> None:
        pass
