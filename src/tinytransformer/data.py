import numpy as np


class CharacterTokenizer:
    def __init__(self, text: str):
        self.characters = sorted(set(text))
        self.character_to_id = {
            character: index for index, character in enumerate(self.characters)
        }
        self.id_to_character = {
            index: character for index, character in enumerate(self.characters)
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> np.ndarray:
        return np.array(
            [self.character_to_id[character] for character in text],
            dtype=np.int64,
        )

    def decode(self, token_ids) -> str:
        return "".join(self.id_to_character[int(token_id)] for token_id in token_ids)


def create_batch(
    encoded_text: np.ndarray,
    batch_size: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:

    maximum_start = len(encoded_text) - sequence_length - 1

    # gives us a list random starting indexes thats valid
    starting_positions = np.random.randint(
        0,
        maximum_start + 1,
        size=batch_size,
    )

    inputs = np.stack(
        [encoded_text[start : start + sequence_length] for start in starting_positions]
    )

    targets = np.stack(
        [
            encoded_text[start + 1 : start + sequence_length + 1]
            for start in starting_positions
        ]
    )

    return inputs, targets
