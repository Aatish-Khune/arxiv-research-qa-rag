"""Paired bootstrap confidence intervals for retriever differences.

Reads the saved per-query ranks (reports/per_query_ranks.csv), so no model re-runs are needed.
Each bootstrap sample resamples the SAME queries for both retrievers (paired), which is the
right test when every system is evaluated on one shared query set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paperqa.config import REPORTS_DIR, SEED

BASELINE = "bm25"


def hits_at_k(ranks: pd.Series, k: int) -> np.ndarray:
    """1.0 where the relevant paper is in the top k (NaN rank = not retrieved)."""
    return ((ranks <= k) & ranks.notna()).to_numpy(dtype=float)


def reciprocal_ranks(ranks: pd.Series, k: int = 10) -> np.ndarray:
    r = ranks.to_numpy(dtype=float)
    return np.where(np.isfinite(r) & (r <= k), 1.0 / np.where(np.isfinite(r), r, 1.0), 0.0)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                     seed: int = SEED) -> tuple[float, float, float]:
    """Mean of (b - a) and its 95% percentile confidence interval."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run() -> pd.DataFrame:
    ranks = pd.read_csv(REPORTS_DIR / "per_query_ranks.csv")
    systems = [c.removeprefix("rank_") for c in ranks.columns if c.startswith("rank_")]
    rows = []
    for system in systems:
        if system == BASELINE:
            continue
        for metric, fn in (("recall@1", lambda r: hits_at_k(r, 1)),
                           ("mrr@10", reciprocal_ranks)):
            delta, lo, hi = paired_bootstrap(fn(ranks[f"rank_{BASELINE}"]),
                                             fn(ranks[f"rank_{system}"]))
            rows.append({"system": system, "vs": BASELINE, "metric": metric,
                         "delta": round(delta, 3), "ci95_low": round(lo, 3),
                         "ci95_high": round(hi, 3),
                         "significant": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "significance.csv", index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
