# TinyTransformer

A tiny causal transformer built from first principles with MiniNN, TensorGrad,
and NumPy.

This project is the fourth step in a learning-first path toward a small
Dreamer-inspired world model. It will implement and verify:

- token embeddings and positional information;
- layer normalization;
- causal self-attention;
- residual connections and feed-forward layers;
- a complete transformer block;
- next-token training and an explicit causality test.

The project favors small, readable implementations, hand-worked tensor shapes,
and focused gradient tests over scale or performance.

## Development

Create a virtual environment and install the project in editable mode:

```bash
uv venv
uv pip install --python .venv/bin/python --editable ../MiniNN --editable .
```

Run the tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
# TinyTransformer
