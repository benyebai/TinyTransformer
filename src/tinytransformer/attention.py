import numpy as np
from mininn import Module, Parameter
from mininn.linear import Linear
from tensorgrad import Tensor, concatenate


# for one head
class CausalSelfAttentionHead(Module):
    def __init__(
        self,
        model_size: int,  # embedding size
        head_size: int,  # number of features we will use for our K,Q,V
    ):
        # will explain in write up the way I intuitively understand it
        self.query = Linear(model_size, head_size)
        self.key = Linear(model_size, head_size)
        self.value = Linear(model_size, head_size)
        self.head_size = head_size

    def __call__(self, inputs: Tensor) -> Tensor:
        # inputs shape: (B, T, model_size)
        # create Q, K, V
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        # calculate scaled attention scores
        K_transposed = K.transpose(0, 2, 1)
        scores = Q @ K_transposed
        scores = scores / np.sqrt(self.head_size)
        # apply causal mask
        sequence_length = inputs.data.shape[1]
        causal_mask = (
            np.triu(
                np.ones((sequence_length, sequence_length)),
                k=1,
            )
            * -1e9
        )
        masked_scores = scores + causal_mask
        # softmax across keys
        attention_weights = masked_scores.softmax(axis=-1)
        # mix values
        output = attention_weights @ V
        # return shape: (B, T, head_size)
        return output

    def parameters(self) -> list[Parameter]:
        # combine parameters from query, key, and value projections
        return self.query.parameters() + self.key.parameters() + self.value.parameters()


class MultiHeadCausalSelfAttention(Module):
    def __init__(self, model_size: int, number_of_heads: int):
        if model_size % number_of_heads != 0:
            raise ValueError("model_size must be divisible by number_of_heads")

        head_size = model_size // number_of_heads
        self.heads = [
            CausalSelfAttentionHead(model_size, head_size)
            for _ in range(number_of_heads)
        ]
        self.output_projection = Linear(model_size, model_size)

    def __call__(self, inputs: Tensor) -> Tensor:
        head_outputs = [head(inputs) for head in self.heads]
        combined = concatenate(head_outputs, axis=-1)
        return self.output_projection(combined)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for head in self.heads:
            parameters += head.parameters()

        return parameters + self.output_projection.parameters()
