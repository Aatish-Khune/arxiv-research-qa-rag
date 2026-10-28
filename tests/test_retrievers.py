import numpy as np
import pytest

from paperqa.answer import extract_evidence, format_answer
from paperqa.ingest import parse_feed
from paperqa.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
    build_chunks,
    rrf_fuse,
    top_k,
)


def _dense(docs, encoder):
    chunks = build_chunks(docs)
    emb = encoder.encode(chunks["text"].tolist())
    return DenseRetriever(chunks["paper_idx"].to_numpy(), emb, encoder)


def test_top_k_orders_and_breaks_ties_by_index():
    assert top_k(np.array([0.1, 0.9, 0.9, 0.5]), 3) == [(1, 0.9), (2, 0.9), (3, 0.5)]


def test_rrf_fuse_math():
    a = [(1, 9.0), (2, 8.0)]
    b = [(2, 0.9), (3, 0.8)]
    fused = dict(rrf_fuse([a, b], k=3, rrf_k=60))
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 in a, rank 1 in b
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[3] == pytest.approx(1 / 62)
    assert rrf_fuse([a, b], k=1)[0][0] == 2  # appears in both lists -> wins


def test_rrf_ignores_raw_score_scale():
    a = [(1, 1e6), (2, 1e-6)]
    b = [(1, 0.2), (2, 0.1)]
    assert [d for d, _ in rrf_fuse([a, b], k=2)] == [1, 2]


@pytest.mark.parametrize("make", ["bm25", "dense", "hybrid", "rerank"])
def test_each_retriever_finds_the_known_item(make, docs, encoder, cross_encoder):
    bm25 = BM25Retriever(docs)
    dense = _dense(docs, encoder)
    hybrid = HybridRetriever(bm25, dense)
    retriever = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "rerank": RerankedRetriever(hybrid, docs, cross_encoder),
    }[make]
    hits = retriever.search("diffusion models reverse noising images", k=3)
    assert hits[0][0] == 2
    assert len({d for d, _ in hits}) == len(hits)  # no duplicate papers


def test_dense_scores_paper_by_best_chunk(encoder):
    long_doc = " ".join(["filler words about nothing"] * 60) + " quantum annealing optimizer"
    docs = [long_doc, "unrelated text on cooking recipes"]
    dense = _dense(docs, encoder)
    assert dense.search("quantum annealing optimizer", k=2)[0][0] == 0
    assert build_chunks(docs).groupby("paper_idx").size()[0] > 1  # long doc was chunked


def test_bm25_empty_query_returns_nothing(docs):
    assert BM25Retriever(docs).search("the of and", k=5) == []


def test_extractive_answer_cites_sources(docs, encoder):
    q = encoder.encode(["how do federated learning devices share data"])[0]
    evidence = extract_evidence(q, hits=[3, 0], abstracts=docs, encode=encoder.encode)
    assert evidence[0].source == 1 and evidence[0].paper_idx == 3
    assert format_answer(evidence[:1]).endswith("[1]")
    assert all(e.sentence in docs[e.paper_idx] for e in evidence)  # verbatim, no invention


FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2610.01234v2</id>
    <published>2026-10-20T17:59:59Z</published>
    <title>A Study of
      Things</title>
    <summary>  We study
      things carefully. </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/><category term="stat.ML"/>
  </entry>
</feed>"""


def test_parse_feed():
    (paper,) = parse_feed(FEED)
    assert paper == {
        "arxiv_id": "2610.01234",
        "title": "A Study of Things",
        "abstract": "We study things carefully.",
        "authors": "Ada Lovelace, Alan Turing",
        "published": "2026-10-20",
        "primary_category": "cs.LG",
        "categories": "cs.LG stat.ML",
    }
