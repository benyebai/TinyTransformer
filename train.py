from pathlib import Path

import numpy as np
from mininn.optimizers import Adam

from tinytransformer.data import CharacterTokenizer, create_batch
from tinytransformer.loss import cross_entropy
from tinytransformer.model import TinyTransformer

text = Path("data/input.txt").read_text()
tokenizer = CharacterTokenizer(text)
encoded_text = tokenizer.encode(text)

model = TinyTransformer(
    vocabulary_size=tokenizer.vocabulary_size,
    model_size=64,
    number_of_heads=4,
    number_of_layers=4,
    max_sequence_length=128,
)

optimizer = Adam(
    model.parameters(),
    learning_rate=1e-3,
)

for step in range(10000):
    inputs, targets = create_batch(
        encoded_text,
        batch_size=8,
        sequence_length=128,
    )

    model.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step}: loss {loss.data.item():.4f}")
