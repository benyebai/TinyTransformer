import unittest

import numpy as np
from tensorgrad import Tensor

from tinytransformer.block import TransformerBlock


class TestTransformerBlock(unittest.TestCase):
    def test_forward_preserves_input_shape(self):
        block = TransformerBlock(model_size=4, number_of_heads=2)
        inputs = Tensor(np.random.randn(3, 5, 4))

        output = block(inputs)

        self.assertEqual(output.data.shape, inputs.data.shape)

    def test_parameters_include_every_component(self):
        block = TransformerBlock(model_size=4, number_of_heads=2)

        expected = (
            block.norm_before_attention.parameters()
            + block.attention.parameters()
            + block.norm_before_feed_forward.parameters()
            + block.feed_forward_expansion.parameters()
            + block.feed_forward_projection.parameters()
        )

        self.assertEqual(block.parameters(), expected)

    def test_residual_connections_preserve_input_when_branches_are_zero(self):
        block = TransformerBlock(model_size=4, number_of_heads=2)
        block.attention.output_projection.weight.data[...] = 0.0
        block.attention.output_projection.bias.data[...] = 0.0
        block.feed_forward_projection.weight.data[...] = 0.0
        block.feed_forward_projection.bias.data[...] = 0.0
        inputs = Tensor(np.random.randn(2, 3, 4))

        output = block(inputs)

        np.testing.assert_allclose(output.data, inputs.data)

    def test_changing_future_token_does_not_change_earlier_outputs(self):
        np.random.seed(19)
        block = TransformerBlock(model_size=4, number_of_heads=2)
        original = Tensor(
            [[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]]
        )
        changed_future = Tensor(
            [[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [-9.0, 4.0, 12.0, 6.0]]]
        )

        original_output = block(original)
        changed_output = block(changed_future)

        np.testing.assert_allclose(
            original_output.data[:, :2],
            changed_output.data[:, :2],
        )

    def test_backward_reaches_input_and_produces_finite_parameter_gradients(self):
        np.random.seed(23)
        block = TransformerBlock(model_size=4, number_of_heads=2)
        inputs = Tensor(np.random.randn(2, 3, 4))

        block(inputs).sum().backward()

        self.assertTrue(np.all(np.isfinite(inputs.grad)))
        self.assertTrue(np.any(inputs.grad != 0.0))
        for parameter in block.parameters():
            self.assertTrue(np.all(np.isfinite(parameter.grad)))


if __name__ == "__main__":
    unittest.main()
