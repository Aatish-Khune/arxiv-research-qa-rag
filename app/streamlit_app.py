"""Streamlit demo: ask a question about recent ML research, get a cited extractive answer.

Run locally with `python run.py app`.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paperqa.answer import extract_evidence  # noqa: E402
from paperqa.config import REPORTS_DIR  # noqa: E402
from paperqa.index import load_papers, load_retrievers  # noqa: E402

METHODS = {
    "hybrid_rerank": "Hybrid + cross-encoder rerank (best)",
    "hybrid_rrf": "Hybrid (BM25 + dense, RRF)",
    "dense": "Dense embeddings (MiniLM + FAISS)",
    "bm25": "BM25 keyword search (baseline)",
}
EXAMPLES = [
    "How can large language models be made more robust to jailbreak attacks?",
    "Methods for training graph neural networks on very large graphs",
    "Does data augmentation help diffusion models with small datasets?",
    "Federated learning with non-IID clients",
]

st.set_page_config(page_title="arXiv ML Research Q&A", layout="wide")


@st.cache_resource(show_spinner="Loading the corpus and models (first run downloads ~180 MB)...")
def resources():
    return load_papers(), load_retrievers(rerank=True)


papers, retrievers = resources()
abstracts = papers["abstract"].tolist()
info_path = REPORTS_DIR / "eval_info.json"
info = json.loads(info_path.read_text()) if info_path.exists() else {}

st.title("Ask recent machine-learning research")
st.caption(
    f"Searches {len(papers):,} cs.LG arXiv abstracts"
    + (f" ({info['date_range'][0]} to {info['date_range'][1]})" if info else "")
    + ". Answers are extracted sentences from the cited papers, never generated text."
)

with st.sidebar:
    method = st.radio("Retrieval method", list(METHODS), format_func=METHODS.get)
    k = st.slider("Papers to retrieve", 3, 10, 5)
    st.markdown("**Try an example**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    summary = REPORTS_DIR / "summary.md"
    if summary.exists():
        with st.expander("Evaluation results (500 queries)"):
            st.markdown(summary.read_text(encoding="utf-8"))

question = st.text_input("Your question", key="q", placeholder=EXAMPLES[0])
if not question.strip():
    st.info("Type a question or pick an example from the sidebar.")
    st.stop()

with st.spinner("Searching..."):
    hits = retrievers[method].search(question, k)
    dense = retrievers["dense"]
    evidence = extract_evidence(dense.encode([question])[0], [d for d, _ in hits], abstracts,
                                dense.encode)

if not hits:
    st.warning("No results. Try more specific words.")
    st.stop()

st.subheader("Answer")
st.markdown(" ".join(f"{html.escape(e.sentence)} **[{e.source}]**" for e in evidence))
st.caption("Each sentence is quoted verbatim from the numbered source below.")

st.subheader("Sources")
quoted = {(e.paper_idx, e.sentence) for e in evidence}
for n, (paper_idx, score) in enumerate(hits, start=1):
    p = papers.iloc[paper_idx]
    authors = p["authors"] if len(p["authors"]) < 120 else p["authors"][:117] + "..."
    st.markdown(
        f"**[{n}] [{html.escape(p['title'])}](https://arxiv.org/abs/{p['arxiv_id']})**  \n"
        f"{html.escape(authors)} · {p['published']} · score {score:.3f}"
    )
    with st.expander("Abstract"):
        text = html.escape(p["abstract"])
        for idx, sentence in quoted:
            if idx == paper_idx:
                text = text.replace(html.escape(sentence), f"**{html.escape(sentence)}**")
        st.markdown(text)
