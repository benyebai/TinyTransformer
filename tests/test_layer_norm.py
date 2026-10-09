import unittest

import numpy as np
from tensorgrad import Tensor

from tinytransformer.layer_norm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_parameters_start_as_identity_scale_and_zero_shift(self):
        layer = LayerNorm(normalized_size=3)

        self.assertEqual(layer.parameters(), [layer.gamma, layer.beta])
        np.testing.assert_array_equal(layer.gamma.data, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(layer.beta.data, [0.0, 0.0, 0.0])

    def test_forward_matches_numpy_per_token(self):
        data = np.array(
            [
                [[1.0, 2.0, 3.0], [10.0, 14.0, 18.0]],
                [[-2.0, 1.0, 7.0], [5.0, 5.0, 8.0]],
            ]
        )
        layer = LayerNorm(normalized_size=3, epsilon=1e-5)

        output = layer(Tensor(data))

        mean = data.mean(axis=-1, keepdims=True)
        variance = ((data - mean) ** 2).mean(axis=-1, keepdims=True)
        expected = (data - mean) / np.sqrt(variance + 1e-5)
        self.assertEqual(output.data.shape, data.shape)
        np.testing.assert_allclose(output.data, expected)

    def test_gamma_and_beta_scale_each_feature(self):
        layer = LayerNorm(normalized_size=3)
        layer.gamma.data[...] = [2.0, 0.5, -1.0]
        layer.beta.data[...] = [0.3, -0.2, 1.0]
        inputs = Tensor([[1.0, 2.0, 3.0]])

        output = layer(inputs)

        data = inputs.data
        mean = data.mean(axis=-1, keepdims=True)
        variance = ((data - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (data - mean) / np.sqrt(variance + layer.epsilon)
        expected = normalized * layer.gamma.data + layer.beta.data
        np.testing.assert_allclose(output.data, expected)

    def test_constant_input_remains_finite(self):
        layer = LayerNorm(normalized_size=4)

        output = layer(Tensor([[5.0, 5.0, 5.0, 5.0]]))

        np.testing.assert_array_equal(output.data, [[0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(output.data).all())

    def test_backward_matches_central_finite_differences(self):
        layer = LayerNorm(normalized_size=3, epsilon=1e-5)
        layer.gamma.data[...] = [1.2, 0.7, -0.4]
        layer.beta.data[...] = [0.1, -0.2, 0.3]
        inputs = Tensor([[1.0, -2.0, 4.0], [0.5, 3.0, -1.0]])
        weights = Tensor([[0.3, -0.8, 1.1], [-0.4, 0.2, 0.9]])

        (layer(inputs) * weights).sum().backward()

        analytical = [
            inputs.grad.copy(),
            layer.gamma.grad.copy(),
            layer.beta.grad.copy(),
        ]
        values = [inputs.data, layer.gamma.data, layer.beta.data]
        numerical = []
        epsilon = 1e-6

        for value in values:
            gradient = np.zeros_like(value)
            for index in np.ndindex(value.shape):
                original = value[index]
                value[index] = original + epsilon
                plus = (layer(inputs) * weights).sum().data.item()
                value[index] = original - epsilon
                minus = (layer(inputs) * weights).sum().data.item()
                value[index] = original
                gradient[index] = (plus - minus) / (2.0 * epsilon)
            numerical.append(gradient)

        for actual, expected in zip(analytical, numerical):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
