"""Four retrieval configurations over the abstract corpus, sharing one interface:
`search(query, k) -> list[(paper_index, score)]`, best first.

1. BM25Retriever       - lexical baseline (rank-bm25)
2. DenseRetriever      - MiniLM sentence embeddings + FAISS, paper score = best chunk
3. HybridRetriever     - reciprocal-rank fusion (RRF) of BM25 and dense rankings
4. RerankedRetriever   - cross-encoder rescoring of the hybrid top candidates
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from paperqa.config import CANDIDATES, CHUNK_OVERLAP, CHUNK_WORDS, RERANK_DEPTH, RRF_K
from paperqa.text import chunk_words, tokenize

Ranking = list[tuple[int, float]]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> Ranking: ...


def top_k(scores: np.ndarray, k: int) -> Ranking:
    """Indices of the k highest scores, best first (stable for ties)."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return [(int(i), float(scores[i])) for i in idx]


def rrf_fuse(rankings: Sequence[Ranking], k: int, rrf_k: int = RRF_K) -> Ranking:
    """Reciprocal-rank fusion: score(d) = sum over rankings of 1 / (rrf_k + rank(d)).

    Uses only ranks, so BM25 and cosine scores never need to be put on the same scale.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:k]


def build_chunks(abstracts: Sequence[str]) -> pd.DataFrame:
    """One row per chunk: which paper it came from and its text."""
    rows = [
        {"paper_idx": i, "text": chunk}
        for i, abstract in enumerate(abstracts)
        for chunk in chunk_words(abstract, CHUNK_WORDS, CHUNK_OVERLAP)
    ]
    return pd.DataFrame(rows)


class BM25Retriever:
    name = "bm25"

    def __init__(self, documents: Sequence[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def search(self, query: str, k: int) -> Ranking:
        tokens = tokenize(query)
        if not tokens:
            return []
        return top_k(np.asarray(self.bm25.get_scores(tokens)), k)


class DenseRetriever:
    """Exact inner-product search over L2-normalized chunk embeddings (= cosine similarity).
    A paper's score is its best-matching chunk."""

    name = "dense"

    def __init__(self, chunk_paper_idx: np.ndarray, chunk_embeddings: np.ndarray, encoder) -> None:
        import faiss

        self.chunk_paper_idx = np.asarray(chunk_paper_idx)
        emb = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.encoder = encoder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encoder.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=64
        ).astype(np.float32)

    def search(self, query: str, k: int) -> Ranking:
        return self.search_vector(self.encode([query])[0], k)

    def search_vector(self, qvec: np.ndarray, k: int) -> Ranking:
        n_chunks = min(self.index.ntotal, k * 4)  # over-fetch chunks, then dedupe by paper
        sims, ids = self.index.search(qvec[None, :].astype(np.float32), n_chunks)
        best: dict[int, float] = {}
        for sim, cid in zip(sims[0], ids[0], strict=True):
            if cid < 0:
                continue
            paper = int(self.chunk_paper_idx[cid])
            if paper not in best:  # FAISS returns chunks best-first
                best[paper] = float(sim)
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:k]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever) -> None:
        self.bm25, self.dense = bm25, dense

    def search(self, query: str, k: int) -> Ranking:
        return rrf_fuse(
            [self.bm25.search(query, CANDIDATES), self.dense.search(query, CANDIDATES)], k
        )


class RerankedRetriever:
    """Rescore the hybrid top `RERANK_DEPTH` (query, abstract) pairs with a cross-encoder,
    which reads query and document together (slower, but more accurate than embeddings)."""

    name = "hybrid_rerank"

    def __init__(self, hybrid: HybridRetriever, documents: Sequence[str], cross_encoder) -> None:
        self.hybrid, self.documents, self.cross_encoder = hybrid, documents, cross_encoder

    def search(self, query: str, k: int) -> Ranking:
        candidates = self.hybrid.search(query, max(k, RERANK_DEPTH))
        head, tail = candidates[:RERANK_DEPTH], candidates[RERANK_DEPTH:]
        if not head:
            return []
        scores = self.cross_encoder.predict(
            [(query, self.documents[doc]) for doc, _ in head], batch_size=32
        )
        reranked = sorted(
            zip([doc for doc, _ in head], np.asarray(scores, dtype=float), strict=True),
            key=lambda item: -item[1],
        )
        return [(int(d), float(s)) for d, s in reranked + tail][:k]
