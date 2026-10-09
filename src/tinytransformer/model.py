import numpy as np
from mininn import Module, Parameter
from mininn.linear import Linear
from tensorgrad import Tensor

from tinytransformer.block import TransformerBlock
from tinytransformer.embedding import Embedding
from tinytransformer.layer_norm import LayerNorm


class TinyTransformer(Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_size: int,
        number_of_heads: int,
        number_of_layers: int,
        max_sequence_length: int,
    ):
        self.max_sequence_length = max_sequence_length
        self.token_embedding = Embedding(vocabulary_size, model_size)
        self.position_embedding = Embedding(max_sequence_length, model_size)
        self.blocks = [
            TransformerBlock(model_size, number_of_heads)
            for _ in range(number_of_layers)
        ]
        self.final_norm = LayerNorm(model_size)
        self.language_model_head = Linear(model_size, vocabulary_size)

    def __call__(self, token_ids: np.ndarray) -> Tensor:
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")

        sequence_length = token_ids.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length")

        positions = np.arange(sequence_length)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        return self.language_model_head(hidden)

    def parameters(self) -> list[Parameter]:
        parameters = (
            self.token_embedding.parameters()
            + self.position_embedding.parameters()
        )

        for block in self.blocks:
            parameters += block.parameters()

        return (
            parameters
            + self.final_norm.parameters()
            + self.language_model_head.parameters()
        )
