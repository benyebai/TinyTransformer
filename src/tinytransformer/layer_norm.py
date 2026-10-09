import numpy as np
from mininn.module import Module
from mininn.parameter import Parameter
from tensorgrad import Tensor


class LayerNorm(Module):
    def __init__(
        self,
        normalized_size: int,
        epsilon: float = 1e-5,
    ):
        # gamma: Parameter of ones, shape (normalized_size,)
        self.gamma = Parameter(
            np.ones(normalized_size),
            name="gamma",
        )
        # beta: Parameter of zeros, shape (normalized_size,)
        self.beta = Parameter(
            np.zeros(normalized_size),
            name="beta",
        )
        # save epsilon
        self.epsilon = epsilon

    def __call__(self, inputs: Tensor) -> Tensor:
        # calculate mean on axis=-1
        mean = inputs.mean(-1, True)
        # center inputs
        centre = inputs - mean
        # calculate variance on axis=-1
        variance = (centre**2).mean(axis=-1, keepdims=True)
        # normalize
        normalization = centre / (variance + self.epsilon) ** (1 / 2)
        # apply gamma and beta
        return self.gamma * normalization + self.beta

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]
