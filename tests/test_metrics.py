import numpy as np
import pandas as pd
import pytest

from paperqa.metrics import mrr_at_k, rank_of, recall_at_k, summarize_ranks
from paperqa.significance import hits_at_k, paired_bootstrap, reciprocal_ranks


def test_hits_and_reciprocal_ranks_handle_missing():
    ranks = pd.Series([1.0, 3.0, np.nan, 12.0])
    np.testing.assert_array_equal(hits_at_k(ranks, 1), [1, 0, 0, 0])
    np.testing.assert_allclose(reciprocal_ranks(ranks, 10), [1, 1 / 3, 0, 0])


def test_paired_bootstrap_detects_clear_gain_and_no_gain():
    a = np.zeros(200)
    delta, lo, hi = paired_bootstrap(a, np.ones(200), n_boot=500)
    assert delta == 1.0 and lo == 1.0 and hi == 1.0
    _, lo, hi = paired_bootstrap(a, a, n_boot=500)
    assert lo == 0.0 and hi == 0.0


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
