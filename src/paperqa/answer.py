"""Extractive, cited answers: pick the sentences from the retrieved abstracts that best match
the question, and tag each with its source number. Every claim is a verbatim sentence from a
cited paper, so the answer can't contain made-up facts (no generative model involved)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from paperqa.text import split_sentences


@dataclass
class Evidence:
    source: int  # 1-based citation number
    paper_idx: int
    sentence: str
    score: float


def extract_evidence(
    question_vec: np.ndarray,
    hits: Sequence[int],
    abstracts: Sequence[str],
    encode,
    max_sentences: int = 4,
    per_source: int = 2,
) -> list[Evidence]:
    """Rank sentences from the hit abstracts by cosine similarity to the question.

    `encode` maps a list of strings to L2-normalized vectors. At most `per_source`
    sentences come from any single paper, so the answer cites several sources.
    """
    candidates: list[tuple[int, int, str]] = []
    for source, paper in enumerate(hits, start=1):
        for sentence in split_sentences(abstracts[paper]):
            candidates.append((source, paper, sentence))
    if not candidates:
        return []
    vecs = encode([s for _, _, s in candidates])
    sims = vecs @ question_vec
    picked: list[Evidence] = []
    used: dict[int, int] = {}
    for i in np.argsort(-sims):
        source, paper, sentence = candidates[i]
        if used.get(source, 0) >= per_source:
            continue
        picked.append(Evidence(source, paper, sentence, float(sims[i])))
        used[source] = used.get(source, 0) + 1
        if len(picked) == max_sentences:
            break
    return picked


def format_answer(evidence: Sequence[Evidence]) -> str:
    """Join evidence sentences into a short answer with inline [n] citations."""
    return " ".join(f"{e.sentence} [{e.source}]" for e in evidence)
