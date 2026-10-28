"""Tiny deterministic stand-ins for the neural models so tests run offline and fast."""

import zlib

import numpy as np
import pytest

from paperqa.text import tokenize

DIM = 1024  # large enough that hash collisions between test words are unlikely


class FakeEncoder:
    """Bag-of-words hashing encoder with the SentenceTransformer.encode signature."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64,
               **_):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, zlib.crc32(tok.encode()) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class FakeCrossEncoder:
    """Scores a (query, doc) pair by word overlap."""

    def predict(self, pairs, batch_size=32):
        return [len(set(tokenize(q)) & set(tokenize(d))) for q, d in pairs]


@pytest.fixture
def docs():
    return [
        "Graph neural networks learn node representations by message passing over edges.",
        "We study reinforcement learning agents that explore sparse reward environments.",
        "Diffusion models generate images by learning to reverse a noising process.",
        "Federated learning trains models across many devices without sharing raw data.",
        "Large language models can follow instructions after supervised fine-tuning.",
    ]


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def cross_encoder():
    return FakeCrossEncoder()
