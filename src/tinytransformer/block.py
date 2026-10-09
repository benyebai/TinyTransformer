from mininn import Module, Parameter
from mininn.linear import Linear
from tensorgrad import Tensor

from tinytransformer.attention import MultiHeadCausalSelfAttention
from tinytransformer.layer_norm import LayerNorm


class TransformerBlock(Module):
    def __init__(self, model_size: int, number_of_heads: int):
        self.norm_before_attention = LayerNorm(model_size)
        self.attention = MultiHeadCausalSelfAttention(
            model_size,
            number_of_heads,
        )

        self.norm_before_feed_forward = LayerNorm(model_size)
        self.feed_forward_expansion = Linear(
            model_size,
            4 * model_size,
        )
        self.feed_forward_projection = Linear(
            4 * model_size,
            model_size,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        normalized = self.norm_before_attention(inputs)
        attention_output = self.attention(normalized)
        after_attention = inputs + attention_output

        normalized = self.norm_before_feed_forward(after_attention)
        hidden = self.feed_forward_expansion(normalized).silu()
        feed_forward_output = self.feed_forward_projection(hidden)

        return after_attention + feed_forward_output

    def parameters(self) -> list[Parameter]:
        return (
            self.norm_before_attention.parameters()
            + self.attention.parameters()
            + self.norm_before_feed_forward.parameters()
            + self.feed_forward_expansion.parameters()
            + self.feed_forward_projection.parameters()
        )
