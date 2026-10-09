# TinyTransformer Mathematics

Equations, tensor shapes, and gradient contracts are recorded here before each
component is implemented.

## Shared shape notation

- `B`: batch size
- `T`: sequence length
- `V`: vocabulary size
- `D`: model or embedding width
- `H`: number of attention heads
- `Dh`: per-head width, with `D = H * Dh`

## Token embedding

An embedding owns a trainable table `E` with shape `(V, D)`. Integer token IDs
`x` select rows from that table:

```text
x shape:    (B, T)
E shape:    (V, D)
E[x] shape: (B, T, D)
```

Repeated token IDs reuse the same row. Their backward contributions must add
into that row rather than overwrite one another.
