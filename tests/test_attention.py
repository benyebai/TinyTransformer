import unittest

import numpy as np
from tensorgrad import Tensor

from tinytransformer.attention import (
    CausalSelfAttentionHead,
    MultiHeadCausalSelfAttention,
)


class TestCausalSelfAttentionHead(unittest.TestCase):
    def test_parameters_combines_query_key_and_value_parameters(self):
        head = CausalSelfAttentionHead(model_size=4, head_size=2)

        self.assertEqual(
            head.parameters(),
            head.query.parameters()
            + head.key.parameters()
            + head.value.parameters(),
        )

    def test_forward_matches_numpy_calculation(self):
        head = CausalSelfAttentionHead(model_size=2, head_size=2)
        for projection in [head.query, head.key, head.value]:
            projection.weight.data[...] = np.eye(2)
            projection.bias.data[...] = 0.0

        data = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])

        output = head(Tensor(data))

        scores = data @ data.transpose(0, 2, 1) / np.sqrt(2.0)
        mask = np.triu(np.ones((3, 3)), k=1) * -1e9
        masked_scores = scores + mask
        shifted = masked_scores - masked_scores.max(axis=-1, keepdims=True)
        weights = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        expected = weights @ data
        self.assertEqual(output.data.shape, (1, 3, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_changing_future_token_does_not_change_earlier_outputs(self):
        np.random.seed(7)
        head = CausalSelfAttentionHead(model_size=3, head_size=2)
        original = Tensor(
            [[[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 1.0]]]
        )
        changed_future = Tensor(
            [[[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [-9.0, 4.0, 12.0]]]
        )

        original_output = head(original)
        changed_output = head(changed_future)

        np.testing.assert_allclose(
            original_output.data[:, :2],
            changed_output.data[:, :2],
        )

    def test_backward_matches_central_finite_differences(self):
        np.random.seed(11)
        head = CausalSelfAttentionHead(model_size=2, head_size=2)
        inputs = Tensor([[[0.4, -0.7], [1.2, 0.3]]])
        output_weights = Tensor([[[0.5, -0.2], [0.8, 0.4]]])

        (head(inputs) * output_weights).sum().backward()

        trainable_values = [inputs] + head.parameters()
        analytical = [value.grad.copy() for value in trainable_values]
        numerical = []
        epsilon = 1e-6

        for value in trainable_values:
            gradient = np.zeros_like(value.data)
            for index in np.ndindex(value.data.shape):
                original = value.data[index]
                value.data[index] = original + epsilon
                plus = (head(inputs) * output_weights).sum().data.item()
                value.data[index] = original - epsilon
                minus = (head(inputs) * output_weights).sum().data.item()
                value.data[index] = original
                gradient[index] = (plus - minus) / (2.0 * epsilon)
            numerical.append(gradient)

        for actual, expected in zip(analytical, numerical):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class TestMultiHeadCausalSelfAttention(unittest.TestCase):
    def test_rejects_model_size_not_divisible_by_number_of_heads(self):
        with self.assertRaises(ValueError):
            MultiHeadCausalSelfAttention(model_size=5, number_of_heads=2)

    def test_forward_preserves_model_shape(self):
        attention = MultiHeadCausalSelfAttention(model_size=4, number_of_heads=2)
        inputs = Tensor(np.random.randn(3, 5, 4))

        output = attention(inputs)

        self.assertEqual(output.data.shape, (3, 5, 4))

    def test_parameters_include_every_head_and_output_projection(self):
        attention = MultiHeadCausalSelfAttention(model_size=4, number_of_heads=2)

        expected = []
        for head in attention.heads:
            expected += head.parameters()
        expected += attention.output_projection.parameters()

        self.assertEqual(attention.parameters(), expected)

    def test_changing_future_token_does_not_change_earlier_outputs(self):
        np.random.seed(13)
        attention = MultiHeadCausalSelfAttention(model_size=4, number_of_heads=2)
        original = Tensor(
            [[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]]
        )
        changed_future = Tensor(
            [[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [-9.0, 4.0, 12.0, 6.0]]]
        )

        original_output = attention(original)
        changed_output = attention(changed_future)

        np.testing.assert_allclose(
            original_output.data[:, :2],
            changed_output.data[:, :2],
        )

    def test_backward_reaches_inputs_and_all_parameters(self):
        np.random.seed(17)
        attention = MultiHeadCausalSelfAttention(model_size=4, number_of_heads=2)
        inputs = Tensor(np.random.randn(1, 3, 4))

        attention(inputs).sum().backward()

        self.assertTrue(np.any(inputs.grad != 0.0))
        for parameter in attention.parameters():
            self.assertTrue(np.all(np.isfinite(parameter.grad)))

        parameters_with_signal = [
            parameter
            for parameter in attention.parameters()
            if np.any(parameter.grad != 0.0)
        ]
        self.assertGreater(len(parameters_with_signal), 0)


if __name__ == "__main__":
    unittest.main()
