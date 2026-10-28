"""Known-item retrieval evaluation.

Query = a paper's title; the one relevant document = that paper's abstract (titles are not
indexed). Labels come for free, so we can evaluate hundreds of queries honestly without hand
labeling. This is a proxy for real questions, not a substitute (see README Limitations).
"""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from paperqa.config import (
    CANDIDATES,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    EMBED_MODEL,
    MLFLOW_DB,
    N_EVAL_QUERIES,
    REPORTS_DIR,
    RERANK_DEPTH,
    RERANK_MODEL,
    RRF_K,
    SEED,
    TOP_K,
)
from paperqa.index import load_papers, load_retrievers
from paperqa.metrics import rank_of, summarize_ranks
from paperqa.text import tokenize


def title_overlap(title: str, abstract: str) -> float:
    """Share of the title's content words that also appear in the abstract."""
    t = set(tokenize(title))
    return len(t & set(tokenize(abstract))) / len(t) if t else 0.0


def sample_queries(papers: pd.DataFrame, n: int = N_EVAL_QUERIES) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(papers), size=min(n, len(papers)), replace=False))
    queries = papers.iloc[idx][["arxiv_id", "title", "abstract"]].copy()
    queries["paper_idx"] = idx
    queries["overlap"] = [title_overlap(t, a) for t, a in zip(queries.title, queries.abstract,
                                                              strict=True)]
    return queries.reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def run_evaluation() -> pd.DataFrame:
    papers = load_papers()
    queries = sample_queries(papers)
    low_overlap = queries["overlap"] < queries["overlap"].median()
    retrievers = load_retrievers(rerank=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("arxiv-research-qa-rag")
    rows, per_query = [], {"paper_idx": queries["paper_idx"], "overlap": queries["overlap"]}
    for name, retriever in retrievers.items():
        start = time.perf_counter()
        ranks = [
            rank_of([d for d, _ in retriever.search(q, TOP_K)], rel)
            for q, rel in zip(queries["title"], queries["paper_idx"], strict=True)
        ]
        ms_per_query = 1000 * (time.perf_counter() - start) / len(queries)
        per_query[f"rank_{name}"] = ranks
        metrics = summarize_ranks(ranks)
        low = [r for r, lo in zip(ranks, low_overlap, strict=True) if lo]
        high = [r for r, lo in zip(ranks, low_overlap, strict=True) if not lo]
        metrics["recall@5_low_overlap"] = summarize_ranks(low)["recall@5"]
        metrics["recall@5_high_overlap"] = summarize_ranks(high)["recall@5"]
        metrics["ms_per_query"] = ms_per_query
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "retriever": name, "n_papers": len(papers), "n_queries": len(queries),
                "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
                "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
                "rrf_k": RRF_K, "candidates": CANDIDATES, "rerank_depth": RERANK_DEPTH,
            })
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
        rows.append({"retriever": name, **{k: round(v, 3) for k, v in metrics.items()}})
        print(f"{name:<14} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "results.csv", index=False)
    (REPORTS_DIR / "summary.md").write_text(to_markdown(results), encoding="utf-8")
    pd.DataFrame(per_query).to_csv(REPORTS_DIR / "per_query_ranks.csv", index=False)
    with open(REPORTS_DIR / "eval_info.json", "w", encoding="utf-8") as fh:
        json.dump({
            "n_papers": len(papers),
            "date_range": [papers["published"].min(), papers["published"].max()],
            "n_queries": len(queries),
            "overlap_median": float(queries["overlap"].median()),
            "seed": SEED,
        }, fh, indent=2)
    return results


if __name__ == "__main__":
    run_evaluation()
