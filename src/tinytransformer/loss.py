import numpy as np
from tensorgrad import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    # Tartets: [[2, 1, 0], [1, 3, 2],]
    # Logits: [[[0.3, 0.6 ...], [...], [...]], ...]
    targets = np.asarray(targets)

    if logits.data.ndim != 3:
        raise ValueError("logits must have shape (batch_size, sequence_length, vocabulary_size)")
    if targets.shape != logits.data.shape[:2]:
        raise ValueError("targets must match the batch and sequence dimensions of logits")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("targets must contain integer token IDs")
    if targets.size == 0:
        raise ValueError("targets cannot be empty")

    vocabulary_size = logits.data.shape[-1]
    if np.any(targets < 0) or np.any(targets >= vocabulary_size):
        raise ValueError("target token ID is outside the vocabulary")

    batch_size, sequence_length = targets.shape
    batch_indices = np.arange(batch_size)[:, None]
    time_indices = np.arange(sequence_length)[None, :]

    # numpy has some crazy advanced element selection
    # and it broadcasts it for batch and time indicies!!!!
    correct_logits = logits[batch_indices, time_indices, targets]

    # this is actually just simple math trick
    # so softmax is given [a, b, c] -> [exp(a) / sum of exp(x), exp(b) / sum of exp(x), exp(c) / sum of exp(x)]
    # then u want the -log(p(b)) lets say, then just do log rules! and simplify
    # -log(p(b)) = logsumexp(everything) - the correct logit
    log_normalizers = logits.logsumexp(axis=-1)

    return (log_normalizers - correct_logits).mean()
