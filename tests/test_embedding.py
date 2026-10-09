import unittest

import numpy as np

from tinytransformer.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = Embedding(vocab_size=4, embedding_size=3)
        self.embedding.weights.data[...] = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]

    def test_weight_table_has_vocabulary_by_embedding_shape(self):
        embedding = Embedding(vocab_size=5, embedding_size=7)

        self.assertEqual(embedding.weights.data.shape, (5, 7))

    def test_parameters_returns_embedding_table(self):
        self.assertEqual(self.embedding.parameters(), [self.embedding.weights])

    def test_sequence_lookup_selects_rows_in_token_order(self):
        output = self.embedding(np.array([2, 0, 3]))

        self.assertEqual(output.data.shape, (3, 3))
        np.testing.assert_allclose(
            output.data,
            [
                [0.7, 0.8, 0.9],
                [0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2],
            ],
        )

    def test_batch_lookup_preserves_batch_and_sequence_dimensions(self):
        output = self.embedding(np.array([[0, 1, 2], [3, 2, 0]]))

        self.assertEqual(output.data.shape, (2, 3, 3))
        np.testing.assert_allclose(
            output.data,
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[1.0, 1.1, 1.2], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
            ],
        )

    def test_repeated_tokens_accumulate_gradients_into_shared_row(self):
        output = self.embedding(np.array([1, 1, 3]))

        output.sum().backward()

        np.testing.assert_array_equal(
            self.embedding.weights.grad,
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
