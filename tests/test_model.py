import unittest

import numpy as np

from tinytransformer.model import TinyTransformer


class TestTinyTransformer(unittest.TestCase):
    def make_model(self):
        return TinyTransformer(
            vocabulary_size=7,
            model_size=4,
            number_of_heads=2,
            number_of_layers=2,
            max_sequence_length=6,
        )

    def test_forward_returns_one_vocabulary_logit_per_token(self):
        model = self.make_model()
        token_ids = np.array([[0, 1, 2], [3, 4, 5]])

        logits = model(token_ids)

        self.assertEqual(logits.data.shape, (2, 3, 7))

    def test_parameters_include_all_model_components(self):
        model = self.make_model()
        expected = (
            model.token_embedding.parameters()
            + model.position_embedding.parameters()
        )
        for block in model.blocks:
            expected += block.parameters()
        expected += model.final_norm.parameters()
        expected += model.language_model_head.parameters()

        self.assertEqual(model.parameters(), expected)

    def test_rejects_sequence_longer_than_configured_maximum(self):
        model = self.make_model()

        with self.assertRaises(ValueError):
            model(np.zeros((1, 7), dtype=int))

    def test_rejects_token_ids_without_batch_dimension(self):
        model = self.make_model()

        with self.assertRaises(ValueError):
            model(np.array([0, 1, 2]))

    def test_changing_future_token_does_not_change_earlier_logits(self):
        np.random.seed(29)
        model = self.make_model()
        original = np.array([[0, 1, 2, 3]])
        changed_future = np.array([[0, 1, 2, 6]])

        original_logits = model(original)
        changed_logits = model(changed_future)

        np.testing.assert_allclose(
            original_logits.data[:, :3],
            changed_logits.data[:, :3],
        )

    def test_backward_reaches_embeddings_and_model_parameters(self):
        np.random.seed(31)
        model = self.make_model()
        token_ids = np.array([[0, 1, 2], [2, 3, 4]])

        model(token_ids).sum().backward()

        self.assertTrue(np.any(model.token_embedding.weights.grad != 0.0))
        self.assertTrue(np.any(model.position_embedding.weights.grad != 0.0))
        for parameter in model.parameters():
            self.assertTrue(np.all(np.isfinite(parameter.grad)))


if __name__ == "__main__":
    unittest.main()
