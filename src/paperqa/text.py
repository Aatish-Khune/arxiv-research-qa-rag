"""Text utilities: cleaning, BM25 tokenization, chunking, and sentence splitting."""

from __future__ import annotations

import re

# Small English stopword list (avoids an NLTK dependency).
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me more
    most my no nor not now of off on once only or other our ours out over own same she should so
    some such than that the their theirs them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you
    your we our us via also using use used based new paper propose proposed show shows""".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def clean_text(text: str) -> str:
    """Collapse whitespace and newlines (arXiv titles/abstracts are hard-wrapped)."""
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25, with stopwords removed."""
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of `size` words (step = size - overlap)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter for highlighting answer evidence."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]
