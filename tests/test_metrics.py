import pytest

from paperqa.metrics import mrr_at_k, rank_of, recall_at_k, summarize_ranks


def test_rank_of():
    assert rank_of([5, 3, 9], 3) == 2
    assert rank_of([5, 3, 9], 7) is None


def test_recall_at_k_on_toy_ranks():
    ranks = [1, 3, None, 10, 2]
    assert recall_at_k(ranks, 1) == pytest.approx(0.2)
    assert recall_at_k(ranks, 5) == pytest.approx(0.6)
    assert recall_at_k(ranks, 10) == pytest.approx(0.8)


def test_mrr_counts_misses_as_zero_and_respects_cutoff():
    ranks = [1, 2, None, 20]
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 0.5 + 0 + 0) / 4)


def test_summarize_ranks_keys():
    out = summarize_ranks([1, None])
    assert set(out) == {"recall@1", "recall@5", "recall@10", "mrr@10"}
    assert out["mrr@10"] == pytest.approx(0.5)
