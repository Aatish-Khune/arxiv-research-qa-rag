"""Build and load the search index artifacts (corpus, chunks, chunk embeddings)."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from paperqa.config import ARTIFACTS_DIR, EMBED_MODEL, RERANK_MODEL
from paperqa.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
    build_chunks,
)

PAPERS = ARTIFACTS_DIR / "papers.parquet"
CHUNKS = ARTIFACTS_DIR / "chunks.parquet"
EMBEDDINGS = ARTIFACTS_DIR / "chunk_embeddings.npy"


@lru_cache(maxsize=1)
def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def load_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, device="cpu")


def load_papers() -> pd.DataFrame:
    return pd.read_parquet(PAPERS)


def build_index() -> None:
    """Chunk every abstract and embed the chunks. Titles are NOT indexed: in evaluation
    they are the queries, so indexing them would make the task trivial."""
    papers = load_papers()
    chunks = build_chunks(papers["abstract"].tolist())
    start = time.perf_counter()
    emb = load_encoder().encode(
        chunks["text"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True,
    )
    # Only the chunk -> paper mapping is needed at search time (text is re-derivable).
    chunks[["paper_idx"]].to_parquet(CHUNKS, index=False)
    # float16 halves the artifact size; cosine rankings are unaffected in practice.
    np.save(EMBEDDINGS, emb.astype(np.float16))
    print(
        f"Embedded {len(chunks):,} chunks from {len(papers):,} papers "
        f"in {time.perf_counter() - start:.0f}s -> {EMBEDDINGS.name} {emb.shape}"
    )


def load_retrievers(rerank: bool = True) -> dict:
    """All retrieval configurations, keyed by name."""
    papers = load_papers()
    chunks = pd.read_parquet(CHUNKS)
    abstracts = papers["abstract"].tolist()
    bm25 = BM25Retriever(abstracts)
    dense = DenseRetriever(
        chunks["paper_idx"].to_numpy(), np.load(EMBEDDINGS).astype(np.float32), load_encoder()
    )
    hybrid = HybridRetriever(bm25, dense)
    retrievers = {r.name: r for r in (bm25, dense, hybrid)}
    if rerank:
        reranked = RerankedRetriever(hybrid, abstracts, load_cross_encoder())
        retrievers[reranked.name] = reranked
    return retrievers


if __name__ == "__main__":
    build_index()
