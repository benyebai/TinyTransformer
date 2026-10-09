import unittest

from mininn import Module
from tensorgrad import Tensor

import tinytransformer


class TestInstallation(unittest.TestCase):
    def test_dependencies_are_available(self):
        self.assertIsNotNone(tinytransformer)
        self.assertIsNotNone(Module)

        value = Tensor([1.0, 2.0])
        value.sum().backward()
        self.assertEqual(value.grad.tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
