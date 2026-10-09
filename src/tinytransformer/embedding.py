import numpy as np
from mininn.module import Module
from mininn.parameter import Parameter


class Embedding(Module):
    def __init__(self, vocab_size, embedding_size):
        self.weights = Parameter(
            np.random.normal(loc=0.0, scale=0.02, size=(vocab_size, embedding_size)),
            name="weights",
        )

    def __call__(self, token_ids):
        return self.weights[token_ids]

    def parameters(self) -> list[Parameter]:
        return [self.weights]
