"""Paths, data settings, model names, and retrieval defaults."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Keep downloaded Hugging Face models inside the project (gitignored), not in ~/.cache.
# Must be set before sentence-transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
DATA_DIR = ROOT / "data"  # raw API responses (gitignored)
ARTIFACTS_DIR = ROOT / "artifacts"  # cleaned corpus + embeddings, shipped with the app
REPORTS_DIR = ROOT / "reports"
MLFLOW_DB = ROOT / "mlflow.db"

SEED = 42

# arXiv API (descriptive metadata is CC0 under the arXiv API Terms of Use).
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_CATEGORY = "cs.LG"
N_PAPERS = 10_000
PAGE_SIZE = 1_000
REQUEST_DELAY_S = 3.1  # Terms of Use: no more than 1 request every 3 seconds

# Open models from the Hugging Face Hub (free, run locally on CPU).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking: abstracts can exceed MiniLM's 256-token input limit, so split into
# overlapping word windows and score a paper by its best-matching chunk.
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30

# Retrieval / evaluation
RRF_K = 60  # reciprocal-rank-fusion constant (Cormack et al., 2009)
CANDIDATES = 100  # depth retrieved from each retriever before fusion / reranking
RERANK_DEPTH = 30  # hybrid candidates passed to the cross-encoder
N_EVAL_QUERIES = 500
TOP_K = 10
