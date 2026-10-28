"""Retrieval metrics for known-item search (exactly one relevant paper per query)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rank_of(ranking: Sequence[int], relevant: int) -> int | None:
    """1-based rank of the relevant item, or None if it was not retrieved."""
    for i, doc in enumerate(ranking, start=1):
        if doc == relevant:
            return i
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of queries whose relevant item appears in the top k."""
    return float(np.mean([r is not None and r <= k for r in ranks]))


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Mean reciprocal rank, counting 0 when the item is missing from the top k."""
    return float(np.mean([1.0 / r if r is not None and r <= k else 0.0 for r in ranks]))


def summarize_ranks(ranks: Sequence[int | None], ks: Sequence[int] = (1, 5, 10)) -> dict:
    out = {f"recall@{k}": recall_at_k(ranks, k) for k in ks}
    out[f"mrr@{max(ks)}"] = mrr_at_k(ranks, max(ks))
    return out
