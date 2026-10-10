import numpy as np
from tensorgrad import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    # Tartets: [[2, 1, 0], [1, 3, 2],]
    # Logits: [[[0.3, 0.6 ...], [...], [...]], ...]
    targets = np.asarray(targets)

    batch_size, sequence_length = targets.shape
    batch_indices = np.arange(batch_size)[:, None]
    time_indices = np.arange(sequence_length)[None, :]

    # numpy has some crazy advanced element selection
    # and it broadcasts it for batch and time indicies!!!!
    correct_logits = logits[batch_indices, time_indices, targets]

    # this is actually just simple math trick
    # so softmax is given [a, b, c] -> [exp(a) / sum of exp(x), exp(b) / sum of exp(x), exp(c) / sum of exp(x)]
    # then u want the -log(p(b)) lets say, then just do log rules! and simplify
    # -log(p(b)) = logsumexp(everything) - p(b)
    log_normalizers = logits.logsumexp(axis=-1)

    return (log_normalizers - correct_logits).mean()
