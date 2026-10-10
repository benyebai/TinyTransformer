import unittest

import numpy as np
from tensorgrad import Tensor

from tinytransformer.loss import cross_entropy


class TestCrossEntropy(unittest.TestCase):
    def test_equal_logits_have_log_vocabulary_size_loss(self):
        logits = Tensor(np.zeros((2, 3, 4)))
        targets = np.array([[0, 1, 2], [3, 2, 1]])

        loss = cross_entropy(logits, targets)

        self.assertEqual(loss.data.shape, ())
        np.testing.assert_allclose(loss.data, np.log(4.0))

    def test_forward_matches_numpy_calculation(self):
        data = np.array(
            [[[2.0, 3.0, 1.0], [0.5, 1.0, 2.5]]]
        )
        targets = np.array([[1, 2]])

        loss = cross_entropy(Tensor(data), targets)

        maximum = data.max(axis=-1, keepdims=True)
        log_normalizers = (
            maximum + np.log(np.exp(data - maximum).sum(axis=-1, keepdims=True))
        ).squeeze(-1)
        correct_logits = np.array([[data[0, 0, 1], data[0, 1, 2]]])
        expected = (log_normalizers - correct_logits).mean()
        np.testing.assert_allclose(loss.data, expected)

    def test_backward_is_softmax_minus_one_hot_averaged(self):
        data = np.array(
            [[[0.2, -0.4, 1.1], [1.5, 0.3, -0.7]]]
        )
        targets = np.array([[2, 0]])
        logits = Tensor(data)

        cross_entropy(logits, targets).backward()

        shifted = data - data.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        expected[0, 0, 2] -= 1.0
        expected[0, 1, 0] -= 1.0
        expected /= targets.size
        np.testing.assert_allclose(logits.grad, expected)

    def test_rejects_targets_with_wrong_shape(self):
        logits = Tensor(np.zeros((2, 3, 4)))

        with self.assertRaises(ValueError):
            cross_entropy(logits, np.zeros((2, 2), dtype=int))

    def test_rejects_target_outside_vocabulary(self):
        logits = Tensor(np.zeros((1, 2, 3)))

        with self.assertRaises(ValueError):
            cross_entropy(logits, np.array([[0, 3]]))


if __name__ == "__main__":
    unittest.main()
